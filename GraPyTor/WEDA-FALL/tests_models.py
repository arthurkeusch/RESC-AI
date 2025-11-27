import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import xgboost as xgb
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df

def build_feature_columns(df):
    exclude_targets = {"activity_code", "trial_has_fall", "is_fall"}
    feature_cols = [c for c in df.columns if c not in exclude_targets and df[c].dtype.kind in "fc"]
    return feature_cols

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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = (y_true == y_pred).mean() if y_true.size else 0.0
    return float(acc)

def load_xgboost_model(path: Path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, xgb.Booster):
        return model
    if hasattr(model, "get_booster"):
        return model.get_booster()
    raise TypeError(f"Modèle non compatible XGBoost: {path}")

def eval_activity_model(name, booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes):
    print(f"=== Évaluation modèle XGBoost activité : {name} ===")
    dtest = xgb.DMatrix(X_test)
    probs = booster.predict(dtest)
    y_pred_ac = np.argmax(probs, axis=1)

    mask_valid = y_test_ac >= 0
    row_acc = per_row_metrics(y_test_ac[mask_valid], y_pred_ac[mask_valid])

    df = test_df.copy()
    df["pred_ac_id"] = y_pred_ac
    df["pred_activity_code"] = df["pred_ac_id"].map(id_to_ac)

    y_true_trials = []
    y_pred_trials = []

    gb = df.groupby(["user_id", "trial_id", "activity_code"], sort=False)
    for (_, _, _), g in tqdm(gb, total=gb.ngroups, desc="Éval activité par essai", unit="trial"):
        true_ac = str(g["activity_code"].iloc[0])
        true_id = ac_to_id.get(true_ac)
        if true_id is None:
            continue
        pred_counts = Counter(g["pred_activity_code"])
        if not pred_counts:
            continue
        pred_ac = pred_counts.most_common(1)[0][0]
        pred_id = ac_to_id.get(pred_ac)
        if pred_id is None:
            continue
        y_true_trials.append(true_id)
        y_pred_trials.append(pred_id)

    y_true_trials = np.asarray(y_true_trials, dtype=int)
    y_pred_trials = np.asarray(y_pred_trials, dtype=int)
    trial_acc = per_row_metrics(y_true_trials, y_pred_trials)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_trials, y_pred_trials):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    print(f"Accuracy par ligne (activity_code) : {row_acc:.4f}")
    print(f"Accuracy par essai (activity_code) : {trial_acc:.4f}\n")
    labels = [id_to_ac[i] for i in range(n_classes)]
    return row_acc, trial_acc, cm, labels

def eval_trial_has_fall_model(name, booster, X_test, test_df, y_test_trial):
    print(f"=== Évaluation modèle XGBoost trial_has_fall : {name} ===")
    dtest = xgb.DMatrix(X_test)
    probs = booster.predict(dtest)
    y_pred = (probs >= 0.5).astype(int)

    row_acc = per_row_metrics(y_test_trial, y_pred)

    df = test_df.copy()
    df["true_trial"] = y_test_trial
    df["pred_trial"] = y_pred

    y_true_trials = []
    y_pred_trials = []

    gb = df.groupby(["user_id", "trial_id", "activity_code"], sort=False)
    for (_, _, _), g in tqdm(gb, total=gb.ngroups, desc="Éval trial_has_fall par essai", unit="trial"):
        true_val = int(g["true_trial"].iloc[0])
        pred_counts = Counter(g["pred_trial"])
        if not pred_counts:
            continue
        pred_val = pred_counts.most_common(1)[0][0]
        y_true_trials.append(true_val)
        y_pred_trials.append(pred_val)

    y_true_trials = np.asarray(y_true_trials, dtype=int)
    y_pred_trials = np.asarray(y_pred_trials, dtype=int)
    trial_acc = per_row_metrics(y_true_trials, y_pred_trials)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true_trials, y_pred_trials):
        if 0 <= t < 2 and 0 <= p < 2:
            cm[t, p] += 1

    print(f"Accuracy par ligne (trial_has_fall) : {row_acc:.4f}")
    print(f"Accuracy par essai (trial_has_fall) : {trial_acc:.4f}\n")
    return row_acc, trial_acc, cm

def eval_is_fall_model(name, booster, X_test, test_df, y_test_isfall):
    print(f"=== Évaluation modèle XGBoost is_fall : {name} ===")
    dtest = xgb.DMatrix(X_test)
    probs = booster.predict(dtest)
    y_pred = (probs >= 0.5).astype(int)
    row_acc = per_row_metrics(y_test_isfall, y_pred)

    df = test_df.copy()
    df["pred_isfall"] = y_pred

    tp = ((df["is_fall"] == 1) & (df["pred_isfall"] == 1)).sum()
    tn = ((df["is_fall"] == 0) & (df["pred_isfall"] == 0)).sum()
    fp = ((df["is_fall"] == 0) & (df["pred_isfall"] == 1)).sum()
    fn = ((df["is_fall"] == 1) & (df["pred_isfall"] == 0)).sum()

    total = tp + tn + fp + fn
    p = tp + fn
    n = tn + fp

    fp_rate_all = (fp / total * 100) if total > 0 else 0.0
    fn_rate_all = (fn / total * 100) if total > 0 else 0.0
    fpr = (fp / n * 100) if n > 0 else 0.0
    fnr = (fn / p * 100) if p > 0 else 0.0

    print(f"Accuracy par ligne (is_fall) : {row_acc:.4f}")
    print(f"Matrice de confusion is_fall (tp, tn, fp, fn) : {int(tp)}, {int(tn)}, {int(fp)}, {int(fn)}")
    print(f"Total échantillons : {int(total)}")
    print(f"Taux de faux positifs sur tout le dataset : {fp_rate_all:.2f} %")
    print(f"Taux de faux négatifs sur tout le dataset : {fn_rate_all:.2f} %")
    print(f"FPR (fp / (fp+tn)) : {fpr:.2f} %")
    print(f"FNR (fn / (fn+tp)) : {fnr:.2f} %\n")

    confusion = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total": int(total),
        "accuracy": row_acc,
        "fp_rate_all": fp_rate_all,
        "fn_rate_all": fn_rate_all,
        "fpr": fpr,
        "fnr": fnr,
    }
    return row_acc, confusion

def plot_bar_accuracies(metrics):
    labels = []
    accs = []

    if "activity_code" in metrics:
        labels.append("Activity (essais)")
        accs.append(metrics["activity_code"]["trial"] * 100.0)

    if "trial_has_fall" in metrics:
        labels.append("TrialHasFall (essais)")
        accs.append(metrics["trial_has_fall"]["trial"] * 100.0)

    if "is_fall" in metrics:
        labels.append("IsFall (lignes)")
        accs.append(metrics["is_fall"]["line"] * 100.0)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, accs, width)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy des modèles XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)

    for bar, val in zip(bars, accs):
        if not np.isnan(val):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()

def _row_normalized_percent(cm):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        perc_row = np.where(row_sums > 0, cm / row_sums * 100.0, 0.0)
    return perc_row

def plot_activity_confusion(cm, labels):
    perc_row = _row_normalized_percent(cm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(perc_row, interpolation="nearest", cmap="coolwarm", vmin=0, vmax=100)

    ax.set_title("Matrice de confusion activity_code\n(% par rapport à l'activité réelle)")
    ax.set_xlabel("Activité prédite")
    ax.set_ylabel("Activité réelle")

    n_classes = len(labels)
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(n_classes):
        for j in range(n_classes):
            val = int(cm[i, j])
            p = perc_row[i, j]
            text_color = "black" if p < 50 else "white"
            ax.text(
                j, i,
                f"{val}\n{p:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pourcentage de la ligne réelle")
    plt.tight_layout()
    plt.show()

def plot_trial_has_fall_confusion(cm):
    perc_row = _row_normalized_percent(cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(perc_row, interpolation="nearest", cmap="coolwarm", vmin=0, vmax=100)

    ax.set_title("Matrice de confusion trial_has_fall\n(% par rapport au type réel d'essai)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (essai sans chute)", "1 (essai avec chute)"])
    ax.set_yticklabels(["0 (essai sans chute)", "1 (essai avec chute)"])

    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            p = perc_row[i, j]
            text_color = "black" if p < 50 else "white"
            ax.text(
                j, i,
                f"{val}\n{p:.2f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pourcentage de la ligne réelle")
    plt.tight_layout()
    plt.show()

def plot_is_fall_confusion(confusion):
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]

    cm = np.array([[tn, fp],
                   [fn, tp]], dtype=float)

    perc_row = _row_normalized_percent(cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(perc_row, interpolation="nearest", cmap="coolwarm", vmin=0, vmax=100)

    ax.set_title("Matrice de confusion is_fall\n(% par rapport à l'état réel du timestamp)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 (non chute)", "1 (chute)"])
    ax.set_yticklabels(["0 (non chute)", "1 (chute)"])

    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            p = perc_row[i, j]
            text_color = "black" if p < 50 else "white"
            ax.text(
                j, i,
                f"{val}\n{p:.2f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pourcentage de la ligne réelle")
    plt.tight_layout()
    plt.show()

def main():
    train_df, test_df = load_data()
    feature_cols = build_feature_columns(train_df)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_ac, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    y_train_trial, y_test_trial, y_train_isfall, y_test_isfall = build_targets(train_df, test_df)
    n_classes = len(ac_to_id)

    if not MODELS_DIR.is_dir():
        print(f"Dossier de modèles introuvable: {MODELS_DIR}")
        return

    model_files = sorted(MODELS_DIR.glob("xgboost_*.pkl"))
    if not model_files:
        print(f"Aucun modèle XGBoost (*.pkl) trouvé dans {MODELS_DIR}")
        return

    metrics = {}

    for model_path in model_files:
        name = model_path.name
        base = name.rsplit(".", 1)[0]
        parts = base.split("_", 1)
        if len(parts) != 2:
            continue
        prefix, suffix = parts[0], parts[1]
        if prefix.lower() != "xgboost":
            continue

        booster = load_xgboost_model(model_path)

        if suffix == "activity_code":
            line_acc, trial_acc, cm_ac, labels_ac = eval_activity_model(
                name, booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes
            )
            metrics["activity_code"] = {
                "line": line_acc,
                "trial": trial_acc,
                "cm": cm_ac,
                "labels": labels_ac,
            }
        elif suffix == "trial_has_fall":
            line_acc, trial_acc, cm_trial = eval_trial_has_fall_model(
                name, booster, X_test, test_df, y_test_trial
            )
            metrics["trial_has_fall"] = {
                "line": line_acc,
                "trial": trial_acc,
                "cm": cm_trial,
            }
        elif suffix == "is_fall":
            line_acc, confusion = eval_is_fall_model(
                name, booster, X_test, test_df, y_test_isfall
            )
            metrics["is_fall"] = {
                "line": line_acc,
                "trial": None,
                "confusion": confusion,
            }
        else:
            print(f"Type de modèle inconnu pour le fichier: {name}")

    plot_bar_accuracies(metrics)

    if "activity_code" in metrics and "cm" in metrics["activity_code"]:
        plot_activity_confusion(metrics["activity_code"]["cm"], metrics["activity_code"]["labels"])

    if "trial_has_fall" in metrics and "cm" in metrics["trial_has_fall"]:
        plot_trial_has_fall_confusion(metrics["trial_has_fall"]["cm"])

    if "is_fall" in metrics and "confusion" in metrics["is_fall"]:
        plot_is_fall_confusion(metrics["is_fall"]["confusion"])

if __name__ == "__main__":
    main()
