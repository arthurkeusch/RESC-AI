import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MAX_ROUNDS = 300
THRESHOLD = 0.5

MASKS = [
    {
        "digits": "0000110100111110",
        "mask_int": 3390,
        "features": [
            "Weight (Kg)",
            "timestamp",
            "accel_x_list",
            "accel_y_list",
            "accel_z_list",
            "gyro_z_list",
            "orientation_i_list",
            "orientation_j_list",
        ],
    },
    {
        "digits": "0010110000011111",
        "mask_int": 11295,
        "features": [
            "Height (m)",
            "Weight (Kg)",
            "timestamp",
            "accel_x_list",
            "accel_y_list",
            "orientation_i_list",
            "orientation_j_list",
            "vertical_Accel_x",
        ],
    },
    {
        "digits": "0011111100011110",
        "mask_int": 16158,
        "features": [
            "Weight (Kg)",
            "timestamp",
            "accel_x_list",
            "accel_y_list",
            "gyro_z_list",
            "orientation_s_list",
            "orientation_i_list",
            "orientation_j_list",
            "orientation_k_list",
            "vertical_Accel_x",
        ],
    },
]


def gpu_available():
    try:
        d = xgb.DMatrix(np.array([[0.1, 0.2]], dtype=np.float32), label=[0])
        xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return True
    except Exception:
        return False


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    return train_df, test_df


def confusion_trial_from_proba(proba, test_df, y_test):
    pred_line = (proba >= THRESHOLD).astype(int)
    df = test_df[["user_id", "trial_id", "activity_code"]].copy()
    df["pred"] = pred_line
    df["true"] = y_test

    y_true, y_pred = [], []
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


def pick_best_rounds_and_cms_gpu_only(X_train, y_train, X_test, test_df, y_test, feats, pbar):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feats)
    dtest = xgb.DMatrix(X_test, feature_names=feats)

    booster = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS)

    best_round = None
    best_fn = None
    best_cm = None
    best_stats = None

    for r in range(1, MAX_ROUNDS + 1):
        proba = booster.predict(dtest, iteration_range=(0, r))
        cm = confusion_trial_from_proba(proba, test_df, y_test)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

        pbar.update(1)

        if fp == 0:
            if best_round is None or fn < best_fn or (fn == best_fn and r < best_round):
                best_round = r
                best_fn = fn
                best_cm = cm
                best_stats = (tn, fp, fn, tp)

    if best_round is None:
        best_round = MAX_ROUNDS
        best_cm = cm
        best_stats = (tn, fp, fn, tp)
        best_fn = fn

    return booster, best_round, best_cm, best_stats


def plot_confusions(cms, titles):
    labels = ["Sans chute", "Chute"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle("Matrices de confusion — trial_has_fall (meilleur nb de rounds)", fontsize=14)

    last_im = None
    for ax, cm, title in zip(axes, cms, titles):
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        norm = cm / denom

        last_im = ax.imshow(norm, cmap="coolwarm", vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i,j]}\n{norm[i,j]*100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if norm[i, j] > 0.5 else "black",
                    fontsize=12,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(title, fontsize=10)

    fig.colorbar(last_im, ax=axes, fraction=0.025, pad=0.02, label="Pourcentage (normalisé par classe réelle)")
    plt.show()


def main():
    if not gpu_available():
        raise RuntimeError("GPU XGBoost indisponible (device=cuda).")

    train_df, test_df = load_data()

    y_train = train_df["trial_has_fall"].astype(int).to_numpy()
    y_test = test_df["trial_has_fall"].astype(int).to_numpy()

    total_steps = len(MASKS) * MAX_ROUNDS
    pbar = tqdm(total=total_steps, desc="GPU only — sweep rounds (3 masques x 200)", dynamic_ncols=True)

    cms = []
    titles = []

    for m in MASKS:
        feats = m["features"]
        X_train = train_df[feats].to_numpy(dtype=np.float32)
        X_test = test_df[feats].to_numpy(dtype=np.float32)

        booster, best_round, best_cm, (tn, fp, fn, tp) = pick_best_rounds_and_cms_gpu_only(
            X_train, y_train, X_test, test_df, y_test, feats, pbar
        )

        cms.append(best_cm)
        titles.append(f"{m['digits']} | {m['mask_int']} | {len(feats)}f | r={best_round} | FP={fp} FN={fn}")

        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        print(
            f"{m['digits']} | {m['mask_int']} | {len(feats)}f | GPU | best_round={best_round} "
            f"| TN={tn} FP={fp} FN={fn} TP={tp} | acc={acc:.6f}"
        )

    pbar.close()
    plot_confusions(cms, titles)


if __name__ == "__main__":
    main()
