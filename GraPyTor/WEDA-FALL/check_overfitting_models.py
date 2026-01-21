import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "xgboost_hyper_weighted.pkl"

THRESHOLD = 0.5

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

WEIGHT_COL = "Weight (Kg)"


def load_booster_pkl(path: Path) -> xgb.Booster:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        return obj.get_booster()
    raise TypeError(f"Modèle non supporté: {path.name}")


def compute_trial_confusion_from_proba(proba: np.ndarray, test_df_meta: pd.DataFrame, y_test: np.ndarray) -> np.ndarray:
    pred_line = (proba >= THRESHOLD).astype(int)

    df = test_df_meta.copy()
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


def eval_variant(booster: xgb.Booster, X_test: np.ndarray, test_df_meta: pd.DataFrame, y_test: np.ndarray) -> np.ndarray:
    dtest = xgb.DMatrix(X_test, feature_names=FEATURES)
    proba = booster.predict(dtest)
    return compute_trial_confusion_from_proba(proba, test_df_meta, y_test)


def plot_3_confusions(cms, titles):
    labels = ["Sans chute", "Chute"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    fig.suptitle("Stress-test sur la feature Weight (Kg) — trial_has_fall", fontsize=14)

    last_im = None
    for ax, cm, title in zip(axes, cms, titles):
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        norm = cm / denom

        last_im = ax.imshow(norm, cmap="coolwarm", vmin=0, vmax=1, aspect="equal")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}\n{norm[i, j]*100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=12,
                )

        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(f"{title}\nTN={tn} FP={fp} FN={fn} TP={tp} | acc={acc:.6f}", fontsize=10)

    fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="Pourcentage (normalisé par classe réelle)")
    plt.show()


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")

    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    y_test = test_df["trial_has_fall"].astype(int).to_numpy()

    meta = test_df[["user_id", "trial_id", "activity_code"]].copy()

    if WEIGHT_COL not in test_df.columns:
        raise KeyError(f"Colonne poids introuvable: {WEIGHT_COL}")

    booster = load_booster_pkl(MODEL_PATH)

    X_base_df = test_df[FEATURES].copy()
    X_base = X_base_df.to_numpy(dtype=np.float32)

    rng = np.random.default_rng(42)

    cms = []
    titles = []

    # 1) Normal
    cms.append(eval_variant(booster, X_base, meta, y_test))
    titles.append("Normal")

    # 2) Shuffle total des poids entre lignes (détruit toute corrélation)
    X_shuffle_df = X_base_df.copy()
    shuffled = X_shuffle_df[WEIGHT_COL].to_numpy(copy=True)
    rng.shuffle(shuffled)
    X_shuffle_df[WEIGHT_COL] = shuffled
    X_shuffle = X_shuffle_df.to_numpy(dtype=np.float32)
    cms.append(eval_variant(booster, X_shuffle, meta, y_test))
    titles.append("Weight shuffled")

    # 3) Bruit +/- [0, 5] (double) sur les poids
    X_noise_df = X_base_df.copy()
    noise = rng.uniform(-2.5, 2.5, size=len(X_noise_df)).astype(np.float32)
    X_noise_df[WEIGHT_COL] = X_noise_df[WEIGHT_COL].astype(np.float32) + noise
    X_noise = X_noise_df.to_numpy(dtype=np.float32)
    cms.append(eval_variant(booster, X_noise, meta, y_test))
    titles.append("Weight + bruit U(-5, +5)")

    plot_3_confusions(cms, titles)


if __name__ == "__main__":
    main()
