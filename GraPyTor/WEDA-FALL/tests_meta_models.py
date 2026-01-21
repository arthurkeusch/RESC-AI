import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")

META_ORDER = ["Age", "Height (m)", "Weight (Kg)", "Gender"]
TASKS = ["activity_code", "trial_has_fall", "is_fall"]

MAX_BOOST_ROUNDS = 300


def get_xgb_device_params():
    import numpy as _np
    import xgboost as _xgb
    try:
        d = _xgb.DMatrix(_np.array([[0.1, 0.2]], dtype=_np.float32), label=[0])
        _xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        print("✅ GPU XGBoost détecté et utilisé")
        return {"tree_method": "hist", "device": "cuda"}
    except Exception as e:
        print("⚠️ GPU indisponible → fallback CPU :", e)
        return {"tree_method": "hist"}


DEVICE_PARAMS = get_xgb_device_params()


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    for df in (train_df, test_df):
        df["activity_code"] = df["activity_code"].astype(str)
        df["user_id"] = df["user_id"].astype(str)
        df["trial_id"] = df["trial_id"].astype(str)
    return train_df, test_df


def all_feature_columns(df):
    exclude = {"activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def coerce_meta_numeric(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in ["Age", "Height (m)", "Weight (Kg)"]:
        if c in train_df.columns:
            train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    if "Gender" in train_df.columns:
        for df in (train_df, test_df):
            g = df["Gender"]
            if not pd.api.types.is_numeric_dtype(g):
                df["Gender"] = g.astype(str).str.strip().str.lower().map(
                    {"m": 1, "male": 1, "man": 1, "h": 1, "homme": 1, "1": 1,
                     "f": 0, "female": 0, "woman": 0, "femme": 0, "0": 0}
                )
            df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")

    return train_df, test_df


def fill_missing_with_train_mean(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cols:
        mean = float(np.nanmean(train_df[c].to_numpy(dtype=np.float64)))
        if not np.isfinite(mean):
            mean = 0.0
        train_df[c] = train_df[c].fillna(mean)
        test_df[c] = test_df[c].fillna(mean)
    return train_df, test_df


def encode_activity(train_df, test_df):
    ac_train = sorted(train_df["activity_code"].unique())
    ac_to_id = {ac: i for i, ac in enumerate(ac_train)}
    id_to_ac = {i: ac for ac, i in ac_to_id.items()}
    y_train = train_df["activity_code"].map(ac_to_id).astype(int).to_numpy()
    y_test = test_df["activity_code"].map(ac_to_id).fillna(-1).astype(int).to_numpy()
    return y_train, y_test, ac_to_id, id_to_ac


def build_targets(train_df, test_df):
    return (
        train_df["trial_has_fall"].astype(int).to_numpy(),
        test_df["trial_has_fall"].astype(int).to_numpy(),
        train_df["is_fall"].astype(int).to_numpy(),
        test_df["is_fall"].astype(int).to_numpy(),
    )


def per_row_acc(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean()) if len(y_true) else 0.0


def bitstring(mask):
    return "".join("1" if (mask >> i) & 1 else "0" for i in range(4))


def selected_columns_keep_all_except_mask(all_cols, train_df, mask):
    drop = []
    for i, name in enumerate(META_ORDER):
        if ((mask >> i) & 1) == 0:
            if name in train_df.columns:
                drop.append(name)
    cols = [c for c in all_cols if c not in drop]
    return cols


def eval_activity_trial_level(y_pred_line, test_df, ac_to_id, id_to_ac):
    df = test_df[["user_id", "trial_id", "activity_code"]].copy()
    df["pred_id"] = np.asarray(y_pred_line, dtype=int)
    df["pred_code"] = df["pred_id"].map(id_to_ac)

    y_true_trials = []
    y_pred_trials = []
    fp = 0
    fn = 0

    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        true_id = ac_to_id[str(ac)]
        pred_id = ac_to_id[Counter(g["pred_code"]).most_common(1)[0][0]]
        y_true_trials.append(true_id)
        y_pred_trials.append(pred_id)
        if true_id != pred_id:
            fp += 1
            fn += 1

    acc = per_row_acc(y_true_trials, y_pred_trials)
    return acc, fp, fn


def eval_trial_has_fall_trial_level(y_pred_line, test_df, y_test_line):
    df = test_df[["user_id", "trial_id", "activity_code"]].copy()
    df["pred"] = y_pred_line
    df["true"] = y_test_line

    y_true = []
    y_pred = []

    for (_, _, _), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        y_true.append(int(g["true"].iloc[0]))
        y_pred.append(int(Counter(g["pred"]).most_common(1)[0][0]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = per_row_acc(y_true, y_pred)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return acc, fp, fn


def eval_isfall_row_level(y_pred, y_test):
    y_test = np.asarray(y_test, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = per_row_acc(y_test, y_pred)
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    return acc, fp, fn


def train_predict_activity(X_train, y_train, X_test, n_classes):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        **DEVICE_PARAMS,
    }
    booster = xgb.train(params, dtrain, num_boost_round=MAX_BOOST_ROUNDS, verbose_eval=False)
    return np.argmax(booster.predict(dtest), axis=1).astype(int)


def train_predict_binary(X_train, y_train, X_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        **DEVICE_PARAMS,
    }
    booster = xgb.train(params, dtrain, num_boost_round=MAX_BOOST_ROUNDS, verbose_eval=False)
    p = booster.predict(dtest)
    return (p >= 0.5).astype(int)


def plot_acc_fp_fn(task, labels, acc_vals, fp_vals, fn_vals):
    x = np.arange(len(labels))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(16, 7))

    bars_acc = ax1.bar(x - width, acc_vals, width, color="#1f77b4", label="Accuracy (%)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(task)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    bars_fp = ax2.bar(x, fp_vals, width, color="#ff7f0e", label="Faux positifs")
    bars_fn = ax2.bar(x + width, fn_vals, width, color="#2ca02c", label="Faux négatifs")
    ax2.set_ylabel("Nombre d'erreurs")

    y1max = max(acc_vals) if len(acc_vals) else 1.0
    y2max = max(max(fp_vals) if fp_vals else 0, max(fn_vals) if fn_vals else 0, 1)
    ax1.set_ylim(0, max(100, y1max * 1.15))
    ax2.set_ylim(0, y2max * 1.25 + 1)

    for b in bars_acc:
        h = float(b.get_height())
        ax1.text(
            b.get_x() + b.get_width() / 2,
            min(h + 1, ax1.get_ylim()[1] * 0.96),
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=True,
        )

    for b in bars_fp:
        h = float(b.get_height())
        ax2.text(
            b.get_x() + b.get_width() / 2,
            min(h + 0.5, ax2.get_ylim()[1] * 0.96),
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=True,
        )

    for b in bars_fn:
        h = float(b.get_height())
        ax2.text(
            b.get_x() + b.get_width() / 2,
            min(h + 0.5, ax2.get_ylim()[1] * 0.96),
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=True,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.04),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


def main():
    train_df, test_df = load_data()
    train_df, test_df = coerce_meta_numeric(train_df, test_df)

    all_cols = all_feature_columns(train_df)

    y_train_ac, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    y_train_trial, y_test_trial, y_train_isfall, y_test_isfall = build_targets(train_df, test_df)

    results = {t: {} for t in TASKS}
    valid_masks = list(range(1, 16))
    pbar = tqdm(total=len(valid_masks) * 3, desc="Entraînements GPU (Meta drop)", unit="step")

    for mask in valid_masks:
        bits = bitstring(mask)
        cols = selected_columns_keep_all_except_mask(all_cols, train_df, mask)

        tr, te = fill_missing_with_train_mean(train_df, test_df, cols)
        X_train = tr[cols].to_numpy(dtype=np.float32)
        X_test = te[cols].to_numpy(dtype=np.float32)

        y_pred_ac = train_predict_activity(X_train, y_train_ac, X_test, len(ac_to_id))
        acc, fp, fn = eval_activity_trial_level(y_pred_ac, test_df, ac_to_id, id_to_ac)
        results["activity_code"][bits] = dict(acc=acc, fp=fp, fn=fn)
        pbar.update(1)

        y_pred_trial = train_predict_binary(X_train, y_train_trial, X_test)
        acc, fp, fn = eval_trial_has_fall_trial_level(y_pred_trial, test_df, y_test_trial)
        results["trial_has_fall"][bits] = dict(acc=acc, fp=fp, fn=fn)
        pbar.update(1)

        y_pred_isfall = train_predict_binary(X_train, y_train_isfall, X_test)
        acc, fp, fn = eval_isfall_row_level(y_pred_isfall, y_test_isfall)
        results["is_fall"][bits] = dict(acc=acc, fp=fp, fn=fn)
        pbar.update(1)

    pbar.close()

    for task in TASKS:
        items = sorted(results[task].items(), key=lambda x: x[1]["acc"], reverse=True)
        labels = [k for k, _ in items]
        acc_vals = [v["acc"] * 100 for _, v in items]
        fp_vals = [v["fp"] for _, v in items]
        fn_vals = [v["fn"] for _, v in items]
        plot_acc_fp_fn(task, labels, acc_vals, fp_vals, fn_vals)


if __name__ == "__main__":
    main()
