import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")

THRESHOLD = 0.5
ROUNDS_DEFAULT = 169

FEATURES = [
    "Weight (Kg)",
    "timestamp",
    "accel_x_list",
    "accel_y_list",
    "accel_z_list",
    "gyro_z_list",
    "orientation_i_list",
    "orientation_j_list",
]

MODELS_DIR = Path("models")

MODEL_DEFAULT_PATH = MODELS_DIR / "xgboost.pkl"
MODEL_WEIGHTED_PATH = MODELS_DIR / "xgboost_weighted.pkl"
MODEL_HYPER_PATH = MODELS_DIR / "xgboost_hyper.pkl"
MODEL_HYPER_WEIGHTED_PATH = MODELS_DIR / "xgboost_hyper_weighted.pkl"


def round_weight_to_half_tens(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").astype(np.float32)
    return (np.round(v / 5.0) * 5.0).astype(np.float32)


def get_device_params():
    try:
        d = xgb.DMatrix(np.array([[0.1, 0.2]], dtype=np.float32), label=[0])
        xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return {"tree_method": "hist", "device": "cuda"}, "GPU"
    except Exception:
        return {"tree_method": "hist", "nthread": 0}, "CPU"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")

    train_df["Weight (Kg)"] = round_weight_to_half_tens(train_df["Weight (Kg)"])
    test_df["Weight (Kg)"] = round_weight_to_half_tens(test_df["Weight (Kg)"])

    return train_df, test_df


def compute_scale_pos_weight(y):
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    return 1.0 if pos == 0 else (neg / max(1, pos))


def compute_trial_confusion_from_proba(proba, test_df, y_test):
    pred_line = (proba >= THRESHOLD).astype(int)

    df = test_df[["user_id", "trial_id", "activity_code"]].copy()
    df["pred"] = pred_line
    df["true"] = y_test

    y_true = []
    y_pred = []
    for (_, _, _), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        y_true.append(int(g["true"].iloc[0]))
        y_pred.append(int(Counter(g["pred"]).most_common(1)[0][0]))

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return np.array([[tn, fp], [fn, tp]], dtype=int)


def compute_trial_confusion(booster, X_test, test_df, y_test, feature_names):
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    proba = booster.predict(dtest)
    return compute_trial_confusion_from_proba(proba, test_df, y_test)


def load_booster_pkl(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        return obj.get_booster()
    raise TypeError(f"Modèle non supporté: {path.name}")


def train_default(train_df, device_params):
    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["trial_has_fall"].astype(int).to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)

    params = {"objective": "binary:logistic", "eval_metric": "logloss", **device_params}
    booster = xgb.train(params, dtrain, num_boost_round=ROUNDS_DEFAULT)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DEFAULT_PATH, "wb") as f:
        pickle.dump(booster, f)

    return booster


def train_weighted_only(train_df, device_params):
    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["trial_has_fall"].astype(int).to_numpy()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)

    spw = float(compute_scale_pos_weight(y_train))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": spw,
        **device_params,
    }
    booster = xgb.train(params, dtrain, num_boost_round=ROUNDS_DEFAULT)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_WEIGHTED_PATH, "wb") as f:
        pickle.dump(booster, f)

    return booster, spw


def plot_4_confusions(cms, titles):
    labels = ["Sans chute", "Chute"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), constrained_layout=True)
    fig.suptitle(
        "Matrices de confusion — trial_has_fall (Default / Pondéré / Hyper / Hyper+Pondéré)",
        fontsize=14,
    )

    last_im = None
    for ax, cm, title in zip(axes.flat, cms, titles):
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        norm = cm / denom

        last_im = ax.imshow(norm, cmap="coolwarm", vmin=0, vmax=1, aspect="equal")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i,j]}\n{norm[i,j]*100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=11,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(title, fontsize=10)

    fig.colorbar(
        last_im,
        ax=axes.ravel().tolist(),
        fraction=0.03,
        pad=0.02,
        label="Pourcentage (normalisé par classe réelle)",
    )
    plt.show()


def run_all():
    device_params, backend = get_device_params()
    train_df, test_df = load_data()

    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df["trial_has_fall"].astype(int).to_numpy()

    cms = []
    titles = []

    booster_default = train_default(train_df, device_params)
    cm_default = compute_trial_confusion(booster_default, X_test, test_df, y_test, FEATURES)
    tn, fp, fn, tp = int(cm_default[0, 0]), int(cm_default[0, 1]), int(cm_default[1, 0]), int(cm_default[1, 1])
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    cms.append(cm_default)
    titles.append(f"Default ({backend}) r={ROUNDS_DEFAULT}\nFP={fp} FN={fn} acc={acc:.4f}")

    booster_weighted, spw = train_weighted_only(train_df, device_params)
    cm_weighted = compute_trial_confusion(booster_weighted, X_test, test_df, y_test, FEATURES)
    tn, fp, fn, tp = int(cm_weighted[0, 0]), int(cm_weighted[0, 1]), int(cm_weighted[1, 0]), int(cm_weighted[1, 1])
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    cms.append(cm_weighted)
    titles.append(f"Pondéré ({backend}) r={ROUNDS_DEFAULT}\nspw={spw:.2f} FP={fp} FN={fn} acc={acc:.4f}")

    try:
        booster_hyper = load_booster_pkl(MODEL_HYPER_PATH)
        cm_hyper = compute_trial_confusion(booster_hyper, X_test, test_df, y_test, FEATURES)
        tn, fp, fn, tp = int(cm_hyper[0, 0]), int(cm_hyper[0, 1]), int(cm_hyper[1, 0]), int(cm_hyper[1, 1])
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        cms.append(cm_hyper)
        titles.append(f"Hyper (loaded)\nFP={fp} FN={fn} acc={acc:.4f}")
    except Exception:
        cm_hyper = np.zeros((2, 2), dtype=int)
        cms.append(cm_hyper)
        titles.append("Hyper (missing)\nmodels/xgboost_hyper.pkl")

    try:
        booster_hyper_w = load_booster_pkl(MODEL_HYPER_WEIGHTED_PATH)
        cm_hyper_w = compute_trial_confusion(booster_hyper_w, X_test, test_df, y_test, FEATURES)
        tn, fp, fn, tp = int(cm_hyper_w[0, 0]), int(cm_hyper_w[0, 1]), int(cm_hyper_w[1, 0]), int(cm_hyper_w[1, 1])
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        cms.append(cm_hyper_w)
        titles.append(f"Hyper+Pondéré (loaded)\nFP={fp} FN={fn} acc={acc:.4f}")
    except Exception:
        cm_hyper_w = np.zeros((2, 2), dtype=int)
        cms.append(cm_hyper_w)
        titles.append("Hyper+Pondéré (missing)\nmodels/xgboost_hyper_weighted.pkl")

    print(f"✅ backend train={backend}")
    print(f"✅ saved: {MODEL_DEFAULT_PATH.resolve()}")
    print(f"✅ saved: {MODEL_WEIGHTED_PATH.resolve()}")
    if MODEL_HYPER_PATH.exists():
        print(f"✅ loaded: {MODEL_HYPER_PATH.resolve()}")
    if MODEL_HYPER_WEIGHTED_PATH.exists():
        print(f"✅ loaded: {MODEL_HYPER_WEIGHTED_PATH.resolve()}")

    plot_4_confusions(cms, titles)


def main():
    run_all()


if __name__ == "__main__":
    main()
