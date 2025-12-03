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


def load_xgboost_model(path: Path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, xgb.Booster):
        return model
    return model.get_booster()


def eval_activity_model(booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = np.argmax(probs, axis=1)

    mask = y_test_ac >= 0
    row_acc = per_row_metrics(y_test_ac[mask], y_pred[mask])

    df = test_df.copy()
    df["pred_ac"] = y_pred
    df["pred_code"] = df["pred_ac"].map(id_to_ac)

    y_true_trials = []
    y_pred_trials = []

    for (_, _, _), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        true = ac_to_id[str(g["activity_code"].iloc[0])]
        pred = Counter(g["pred_code"]).most_common(1)[0][0]
        y_true_trials.append(true)
        y_pred_trials.append(ac_to_id[pred])

    y_true_trials = np.array(y_true_trials)
    y_pred_trials = np.array(y_pred_trials)
    trial_acc = per_row_metrics(y_true_trials, y_pred_trials)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_trials, y_pred_trials):
        cm[t, p] += 1

    return row_acc, trial_acc, cm, list(id_to_ac.values())


def eval_trial(booster, X_test, test_df, y_test):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = (probs >= 0.5).astype(int)

    df = test_df.copy()
    df["pred"] = y_pred
    df["true"] = y_test

    y_true, y_pred_t = [], []
    for (_, _, _), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        y_true.append(int(g["true"].iloc[0]))
        y_pred_t.append(Counter(g["pred"]).most_common(1)[0][0])

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred_t):
        cm[t, p] += 1

    return per_row_metrics(np.array(y_true), np.array(y_pred_t)), cm


def eval_isfall(booster, X_test, test_df, y_test):
    probs = booster.predict(xgb.DMatrix(X_test))
    y_pred = (probs >= 0.5).astype(int)

    df = test_df.copy()
    df["pred"] = y_pred
    df["true"] = y_test

    tn = ((df.true == 0) & (df.pred == 0)).sum()
    fp = ((df.true == 0) & (df.pred == 1)).sum()
    fn = ((df.true == 1) & (df.pred == 0)).sum()
    tp = ((df.true == 1) & (df.pred == 1)).sum()

    cm = np.array([[tn, fp], [fn, tp]])
    return per_row_metrics(y_test, y_pred), cm


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
                vals.append(metrics[model][g]["trial"] * 100)
            else:
                vals.append(metrics[model][g]["line"] * 100)

        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=MODEL_LABELS[i], color=COLOR_MAP[i])

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{bar.get_height():.1f}%",
                ha='center',
                fontsize=9,
                fontweight="bold"
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

        cm = cm_dict[key]
        norm = cm / cm.sum(axis=1, keepdims=True) * 100

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
                    j, i,
                    f"{val}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=font_val,
                    color=color,
                    fontweight="normal"
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

    for path in MODELS_DIR.glob("xgboost*.pkl"):
        name = path.stem

        if "hyper_pond" in name:
            group = "hyper_pond"
            suffix = name.replace("xgboost_hyper_pond_", "")
        elif "hyper" in name:
            group = "hyper"
            suffix = name.replace("xgboost_hyper_", "")
        elif "pond" in name:
            group = "pond"
            suffix = name.replace("xgboost_pond_", "")
        else:
            group = "standard"
            suffix = name.replace("xgboost_", "")

        booster = load_xgboost_model(path)

        if suffix == "activity_code":
            row, trial, cm, lab = eval_activity_model(
                booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes
            )
            metrics[group][suffix] = {"line": row, "trial": trial}
            cms[suffix][group] = cm

        elif suffix == "trial_has_fall":
            acc, cm = eval_trial(booster, X_test, test_df, y_test_trial)
            metrics[group][suffix] = {"line": acc}
            cms[suffix][group] = cm

        elif suffix == "is_fall":
            acc, cm = eval_isfall(booster, X_test, test_df, y_test_isfall)
            metrics[group][suffix] = {"line": acc}
            cms[suffix][group] = cm

    plot_bar_chart(metrics)
    plot_confusions(cms["activity_code"], list(id_to_ac.values()), "Activity Code")
    plot_confusions(cms["trial_has_fall"], ["Sans chute", "Chute"], "Trial Has Fall")
    plot_confusions(cms["is_fall"], ["Non chute", "Chute"], "Is Fall")


if __name__ == "__main__":
    main()
