import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

MODEL_ORDER = ["standard", "pond", "hyper", "hyper_pond"]
MODEL_LABELS = ["Standard", "Pondéré", "Hyper", "Hyper + Pondéré"]
COLOR_MAP = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

TOP_ACTIVITIES = 25


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude_targets = {"activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude_targets and df[c].dtype.kind in "fc"]


def encode_activity(train_df, test_df):
    ac_train = sorted(train_df["activity_code"].unique())
    ac_to_id = {ac: i for i, ac in enumerate(ac_train)}
    id_to_ac = {i: ac for ac, i in ac_to_id.items()}
    train_y_ac = train_df["activity_code"].map(ac_to_id).astype(int).to_numpy()
    test_y_ac = test_df["activity_code"].map(ac_to_id).fillna(-1).astype(int).to_numpy()
    return train_y_ac, test_y_ac, ac_to_id, id_to_ac


def build_targets(train_df, test_df):
    y_train_trial = train_df["trial_has_fall"].astype(int).to_numpy()
    y_test_trial = test_df["trial_has_fall"].astype(int).to_numpy()
    y_train_isfall = train_df["is_fall"].astype(int).to_numpy()
    y_test_isfall = test_df["is_fall"].astype(int).to_numpy()
    return y_train_trial, y_test_trial, y_train_isfall, y_test_isfall


def per_row_metrics(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def parse_model_group_and_suffix(stem: str):
    if "hyper_pond" in stem:
        group = "hyper_pond"
        suffix = stem.replace("xgboost_hyper_pond_", "")
    elif "hyper" in stem:
        group = "hyper"
        suffix = stem.replace("xgboost_hyper_", "")
    elif "pond" in stem:
        group = "pond"
        suffix = stem.replace("xgboost_pond_", "")
    else:
        group = "standard"
        suffix = stem.replace("xgboost_", "")
    return group, suffix


def is_simple_model_file(path: Path):
    stem = path.stem.lower()
    if not stem.startswith("xgboost_"):
        return False
    if "stacking" in stem or "voting" in stem:
        return False
    group, suffix = parse_model_group_and_suffix(stem)
    if group not in set(MODEL_ORDER):
        return False
    if suffix not in {"activity_code", "trial_has_fall", "is_fall"}:
        return False
    return True


def load_xgboost_model(path: Path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, xgb.Booster):
        return model
    if hasattr(model, "get_booster"):
        return model.get_booster()
    raise TypeError(f"Modèle non supporté (pas Booster / pas get_booster): {path.name}")


def eval_activity_model(booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = np.argmax(probs, axis=1)

    mask = y_test_ac >= 0
    row_acc = per_row_metrics(y_test_ac[mask], y_pred[mask])

    df = test_df.copy()
    df["pred_ac"] = y_pred
    df["pred_code_line"] = df["pred_ac"].map(id_to_ac)

    rows = []
    y_true_trials = []
    y_pred_trials = []

    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        true_code = str(ac)
        pred_code = Counter(g["pred_code_line"]).most_common(1)[0][0]
        rows.append({"activity_true": true_code, "activity_pred": str(pred_code)})
        y_true_trials.append(ac_to_id[true_code])
        y_pred_trials.append(ac_to_id[str(pred_code)])

    y_true_trials = np.array(y_true_trials)
    y_pred_trials = np.array(y_pred_trials)
    trial_acc = per_row_metrics(y_true_trials, y_pred_trials)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_trials, y_pred_trials):
        cm[t, p] += 1

    trial_df = pd.DataFrame(rows)
    return row_acc, trial_acc, cm, list(id_to_ac.values()), trial_df


def eval_trial(booster, X_test, test_df, y_test):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = (probs >= 0.5).astype(int)

    df = test_df.copy()
    df["pred_line"] = y_pred
    df["true_line"] = y_test

    rows = []
    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        rows.append(
            {
                "activity_code": str(ac),
                "true": int(g["true_line"].iloc[0]),
                "pred": int(Counter(g["pred_line"]).most_common(1)[0][0]),
            }
        )

    trial_df = pd.DataFrame(rows)
    y_true = trial_df["true"].to_numpy(dtype=int)
    y_pred_t = trial_df["pred"].to_numpy(dtype=int)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred_t):
        cm[t, p] += 1

    return per_row_metrics(y_true, y_pred_t), cm, trial_df


def eval_isfall(booster, X_test, test_df, y_test):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = (probs >= 0.5).astype(int)

    df = test_df.copy()
    df["pred"] = y_pred
    df["true"] = y_test
    df["activity_code"] = df["activity_code"].astype(str)

    tn = ((df.true == 0) & (df.pred == 0)).sum()
    fp = ((df.true == 0) & (df.pred == 1)).sum()
    fn = ((df.true == 1) & (df.pred == 0)).sum()
    tp = ((df.true == 1) & (df.pred == 1)).sum()

    cm = np.array([[tn, fp], [fn, tp]])
    return per_row_metrics(y_test, y_pred), cm, df


def fn_fp_from_binary_df(df, true_col, pred_col, activity_col):
    fp = df[(df[true_col] == 0) & (df[pred_col] == 1)][activity_col].astype(str)
    fn = df[(df[true_col] == 1) & (df[pred_col] == 0)][activity_col].astype(str)
    return Counter(fn.tolist()), Counter(fp.tolist())


def fn_fp_from_multiclass_trials(trial_df):
    m = trial_df[trial_df["activity_true"].astype(str) != trial_df["activity_pred"].astype(str)]
    fn = Counter(m["activity_true"].astype(str).tolist())
    fp = Counter(m["activity_pred"].astype(str).tolist())
    return fn, fp


def top_union_keys(per_model_fnfp, top_n):
    total = Counter()
    for fn, fp in per_model_fnfp.values():
        total.update(fn)
        total.update(fp)
    return [k for k, _ in total.most_common(top_n)]


def compress_counters(fn: Counter, fp: Counter, keep_keys):
    fn2 = Counter({k: fn.get(k, 0) for k in keep_keys})
    fp2 = Counter({k: fp.get(k, 0) for k in keep_keys})
    fn_other = sum(v for k, v in fn.items() if k not in keep_keys)
    fp_other = sum(v for k, v in fp.items() if k not in keep_keys)
    if fn_other or fp_other:
        fn2["OTHER"] += fn_other
        fp2["OTHER"] += fp_other
    return fn2, fp2


def plot_grouped_fn_fp(task_title, per_model_fnfp, top_n=TOP_ACTIVITIES):
    keep = top_union_keys(per_model_fnfp, top_n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(task_title, fontsize=16)

    for ax, mkey, mlabel in zip(axes.flat, MODEL_ORDER, MODEL_LABELS):
        fn, fp = per_model_fnfp.get(mkey, (Counter(), Counter()))
        fn, fp = compress_counters(fn, fp, keep)

        fn_items = [(k, v) for k, v in fn.most_common() if v > 0]
        fp_items = [(k, v) for k, v in fp.most_common() if v > 0]

        fn_labels = [k for k, v in fn_items]
        fn_vals = [v for k, v in fn_items]
        fp_labels = [k for k, v in fp_items]
        fp_vals = [v for k, v in fp_items]

        gap = 1.25
        x_fn = np.arange(len(fn_vals), dtype=float)
        x_fp = np.arange(len(fp_vals), dtype=float) + (len(fn_vals) + gap)

        bars_fn = ax.bar(x_fn, fn_vals)
        bars_fp = ax.bar(x_fp, fp_vals)

        max_val = 1
        if fn_vals:
            max_val = max(max_val, max(fn_vals))
        if fp_vals:
            max_val = max(max_val, max(fp_vals))

        ylim_top = max_val * 1.55 + 2
        ax.set_ylim(0, ylim_top)

        for b, lab, v in zip(bars_fn, fn_labels, fn_vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(b.get_height() + 0.35, ylim_top * 0.92),
                f"{lab}\n({v})",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=True,
            )

        for b, lab, v in zip(bars_fp, fp_labels, fp_vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(b.get_height() + 0.35, ylim_top * 0.92),
                f"{lab}\n({v})",
                ha="center",
                va="bottom",
                fontsize=7,
                clip_on=True,
            )

        sep_x = len(fn_vals) + gap / 2 - 0.5
        ax.axvline(sep_x, linestyle="--", alpha=0.4)

        ax.set_title(mlabel, fontsize=13)
        ax.set_ylabel("Nombre d'erreurs")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.show()


def plot_bar_chart(metrics):
    groups = ["activity_code", "trial_has_fall", "is_fall"]
    group_labels = ["Activity", "TrialHasFall", "IsFall"]
    x = np.arange(len(group_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(MODEL_ORDER):
        vals = []
        for g in groups:
            if g == "activity_code":
                vals.append(metrics.get(model, {}).get(g, {}).get("trial", 0.0) * 100)
            else:
                vals.append(metrics.get(model, {}).get(g, {}).get("line", 0.0) * 100)

        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABELS[i], color=COLOR_MAP[i])

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparaison des 4 variantes XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_confusions(cm_dict, labels, title):
    n = len(labels)

    if n <= 2:
        cell_size = 3.5
        font_val = 18
        font_label = 14
        fig_scale = 1.2
    else:
        cell_size = 1.5
        font_val = 10
        font_label = 10
        fig_scale = 1.2

    fig_w = n * cell_size * fig_scale
    fig_h = n * cell_size * fig_scale * 0.8

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    for ax, key, name in zip(axes.flat, MODEL_ORDER, MODEL_LABELS):
        cm = cm_dict.get(key, np.zeros((n, n), dtype=int))
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        norm = cm / denom * 100

        im = ax.imshow(norm, vmin=0, vmax=100, cmap="coolwarm", aspect="equal")

        ax.set_title(name, fontsize=14)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45 if n > 3 else 0, ha="right", fontsize=font_label)
        ax.set_yticklabels(labels, fontsize=font_label)

        ax.set_xlabel("Prédit", fontsize=font_label)
        ax.set_ylabel("Réel", fontsize=font_label)

        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                pct = norm[i, j]
                color = "white" if pct > 55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=font_val,
                    color=color,
                    fontweight="normal",
                )

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Pourcentage par classe réelle (%)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.show()


def main():
    train_df, test_df = load_data()
    X_test = test_df[build_feature_columns(train_df)].to_numpy(dtype=np.float32)

    _, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    _, y_test_trial, _, y_test_isfall = build_targets(train_df, test_df)
    n_classes = len(ac_to_id)

    metrics = {k: {} for k in MODEL_ORDER}
    cms = {"activity_code": {}, "trial_has_fall": {}, "is_fall": {}}

    fnfp_activity = {k: (Counter(), Counter()) for k in MODEL_ORDER}
    fnfp_trial = {k: (Counter(), Counter()) for k in MODEL_ORDER}
    fnfp_isfall = {k: (Counter(), Counter()) for k in MODEL_ORDER}

    model_files = sorted([p for p in MODELS_DIR.glob("xgboost*.pkl") if is_simple_model_file(p)], key=lambda p: p.name.lower())

    for path in model_files:
        stem = path.stem.lower()
        group, suffix = parse_model_group_and_suffix(stem)

        booster = load_xgboost_model(path)

        if suffix == "activity_code":
            row, trial, cm, lab, trial_df = eval_activity_model(
                booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes
            )
            metrics[group][suffix] = {"line": row, "trial": trial}
            cms[suffix][group] = cm
            fn, fp = fn_fp_from_multiclass_trials(trial_df)
            fnfp_activity[group] = (fn, fp)

        elif suffix == "trial_has_fall":
            acc, cm, trial_df = eval_trial(booster, X_test, test_df, y_test_trial)
            metrics[group][suffix] = {"line": acc}
            cms[suffix][group] = cm
            fn, fp = fn_fp_from_binary_df(trial_df, "true", "pred", "activity_code")
            fnfp_trial[group] = (fn, fp)

        elif suffix == "is_fall":
            acc, cm, row_df = eval_isfall(booster, X_test, test_df, y_test_isfall)
            metrics[group][suffix] = {"line": acc}
            cms[suffix][group] = cm
            fn, fp = fn_fp_from_binary_df(row_df, "true", "pred", "activity_code")
            fnfp_isfall[group] = (fn, fp)

    plot_bar_chart(metrics)
    plot_confusions(cms["activity_code"], list(id_to_ac.values()), "Activity Code")
    plot_confusions(cms["trial_has_fall"], ["Sans chute", "Chute"], "Trial Has Fall")
    plot_confusions(cms["is_fall"], ["Non chute", "Chute"], "Is Fall")

    plot_grouped_fn_fp("Erreurs par activité — Activity Code", fnfp_activity)
    plot_grouped_fn_fp("Erreurs par activité — Trial Has Fall", fnfp_trial)
    plot_grouped_fn_fp("Erreurs par activité — Is Fall", fnfp_isfall)


if __name__ == "__main__":
    main()
