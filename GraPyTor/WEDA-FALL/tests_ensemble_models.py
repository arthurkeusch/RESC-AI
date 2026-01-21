import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

ENSEMBLE_ORDER = ["voting_soft", "voting_hard", "voting_weighted", "stacking"]
ENSEMBLE_LABELS = ["Voting Soft", "Voting Hard", "Voting Pondéré", "Stacking"]
COLOR_MAP = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

VARIANTS = ["standard", "pond", "hyper", "hyper_pond"]

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


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


class VotingEnsemble:
    def __init__(
        self,
        task: str,
        boosters: dict,
        mode: str = "soft",
        threshold: float = 0.5,
        weights: dict | None = None,
        meta: dict | None = None,
    ):
        self.task = task
        self.boosters = boosters
        self.mode = mode
        self.threshold = float(threshold)
        self.weights = weights if weights is not None else {k: 1.0 for k in boosters.keys()}
        self.meta = meta if meta is not None else {}

    def _w(self, k: str):
        return float(self.weights.get(k, 1.0))

    def _predict_proba_activity_weighted(self, X: np.ndarray):
        dm = xgb.DMatrix(X)
        probs_sum = None
        wsum = 0.0
        for k, b in self.boosters.items():
            p = b.predict(dm)
            w = self._w(k)
            if probs_sum is None:
                probs_sum = p * w
            else:
                probs_sum += p * w
            wsum += w
        return probs_sum / max(wsum, 1e-12)

    def _predict_proba_binary_weighted(self, X: np.ndarray):
        dm = xgb.DMatrix(X)
        psum = 0.0
        wsum = 0.0
        for k, b in self.boosters.items():
            p = b.predict(dm)
            w = self._w(k)
            psum += p * w
            wsum += w
        return psum / max(wsum, 1e-12)

    def _hard_vote_multiclass(self, X: np.ndarray):
        dm = xgb.DMatrix(X)
        preds = []
        for _, b in self.boosters.items():
            p = b.predict(dm)
            preds.append(np.argmax(p, axis=1).astype(int))
        preds = np.stack(preds, axis=0)
        return np.apply_along_axis(lambda a: np.bincount(a).argmax(), 0, preds).astype(int)

    def _hard_vote_binary(self, X: np.ndarray):
        dm = xgb.DMatrix(X)
        preds = []
        for _, b in self.boosters.items():
            p = b.predict(dm)
            preds.append((p >= self.threshold).astype(int))
        preds = np.stack(preds, axis=0)
        return (np.sum(preds, axis=0) >= (preds.shape[0] / 2)).astype(int)

    def predict(self, X: np.ndarray):
        if self.mode not in {"soft", "hard", "weighted"}:
            raise ValueError("mode doit être 'soft', 'hard' ou 'weighted'")

        if self.task == "activity_code":
            if self.mode == "hard":
                return self._hard_vote_multiclass(X)
            probs = self._predict_proba_activity_weighted(X)
            return np.argmax(probs, axis=1).astype(int)

        if self.task in {"trial_has_fall", "is_fall"}:
            if self.mode == "hard":
                return self._hard_vote_binary(X)
            p = self._predict_proba_binary_weighted(X)
            return (p >= self.threshold).astype(int)

        raise ValueError(f"Task inconnue: {self.task}")

    def predict_proba(self, X: np.ndarray):
        if self.mode not in {"soft", "hard", "weighted"}:
            raise ValueError("mode doit être 'soft', 'hard' ou 'weighted'")

        if self.task == "activity_code":
            if self.mode == "hard":
                n_classes = int(self.meta.get("n_classes", 0))
                if n_classes <= 0:
                    id_to_ac = self.meta.get("id_to_ac", {})
                    n_classes = len(id_to_ac)
                y = self._hard_vote_multiclass(X)
                out = np.zeros((len(y), n_classes), dtype=np.float32)
                out[np.arange(len(y)), y] = 1.0
                return out
            return self._predict_proba_activity_weighted(X)

        if self.task in {"trial_has_fall", "is_fall"}:
            if self.mode == "hard":
                y = self._hard_vote_binary(X).astype(np.float32)
                return np.vstack([1.0 - y, y]).T
            p = self._predict_proba_binary_weighted(X)
            return np.vstack([1.0 - p, p]).T

        raise ValueError(f"Task inconnue: {self.task}")


class StackingEnsemble:
    def __init__(self, task: str, boosters: dict, meta: dict | None = None):
        self.task = task
        self.boosters = boosters
        self.meta = meta if meta is not None else {}
        self.w = None
        self.W = None
        self.b = 0.0
        self.threshold = 0.5
        self.n_classes = int(self.meta.get("n_classes", 0))

    def _sigmoid(self, z):
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(np.clip(z, -50.0, 50.0))
        return ez / np.sum(ez, axis=1, keepdims=True)

    def _meta_features_binary(self, X):
        dm = xgb.DMatrix(X)
        return np.stack([self.boosters[k].predict(dm) for k in VARIANTS], axis=1).astype(np.float32)

    def _meta_features_multiclass(self, X):
        dm = xgb.DMatrix(X)
        feats = []
        for k in VARIANTS:
            feats.append(self.boosters[k].predict(dm))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def predict_proba(self, X):
        if self.task == "activity_code":
            F = self._meta_features_multiclass(X).astype(np.float64)
            logits = F @ self.W + self.b
            return self._softmax(logits)
        F = self._meta_features_binary(X).astype(np.float64)
        p = self._sigmoid(F @ self.w + self.b)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        if self.task == "activity_code":
            p = self.predict_proba(X)
            return np.argmax(p, axis=1).astype(int)
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)


def eval_activity_preds(y_pred, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes):
    mask = y_test_ac >= 0
    row_acc = per_row_metrics(y_test_ac[mask], y_pred[mask])

    df = test_df.copy()
    df["pred_ac"] = y_pred
    df["pred_code"] = df["pred_ac"].map(id_to_ac)

    rows = []
    y_true_trials = []
    y_pred_trials = []

    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        true_code = str(ac)
        true = ac_to_id[true_code]
        pred_code = Counter(g["pred_code"]).most_common(1)[0][0]
        pred = ac_to_id[pred_code]
        rows.append({"activity_true": true_code, "activity_pred": str(pred_code)})
        y_true_trials.append(true)
        y_pred_trials.append(pred)

    y_true_trials = np.array(y_true_trials, dtype=int)
    y_pred_trials = np.array(y_pred_trials, dtype=int)
    trial_acc = per_row_metrics(y_true_trials, y_pred_trials)

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_trials, y_pred_trials):
        cm[t, p] += 1

    trial_df = pd.DataFrame(rows)
    return row_acc, trial_acc, cm, trial_df


def eval_trial_preds(y_pred, test_df, y_test):
    df = test_df.copy()
    df["pred"] = y_pred
    df["true"] = y_test

    rows = []
    y_true, y_pred_t = [], []
    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        t = int(g["true"].iloc[0])
        p = int(Counter(g["pred"]).most_common(1)[0][0])
        rows.append({"activity_code": str(ac), "true": t, "pred": p})
        y_true.append(t)
        y_pred_t.append(p)

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred_t):
        cm[t, p] += 1

    trial_df = pd.DataFrame(rows)
    return per_row_metrics(np.array(y_true), np.array(y_pred_t)), cm, trial_df


def eval_isfall_preds(y_pred, test_df, y_test):
    df = test_df.copy()
    df["pred"] = np.asarray(y_pred, dtype=int)
    df["true"] = np.asarray(y_test, dtype=int)
    df["activity_code"] = df["activity_code"].astype(str)

    tn = int(((df.true == 0) & (df.pred == 0)).sum())
    fp = int(((df.true == 0) & (df.pred == 1)).sum())
    fn = int(((df.true == 1) & (df.pred == 0)).sum())
    tp = int(((df.true == 1) & (df.pred == 1)).sum())

    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    return per_row_metrics(df["true"].to_numpy(), df["pred"].to_numpy()), cm, df


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

    for ax, mkey, mlabel in zip(axes.flat, ENSEMBLE_ORDER, ENSEMBLE_LABELS):
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


def discover_ensemble_models():
    models = {k: {} for k in ENSEMBLE_ORDER}

    for suffix in ["activity_code", "trial_has_fall", "is_fall"]:
        p = MODELS_DIR / f"xgboost_stacking_{suffix}.pkl"
        if p.exists():
            models["stacking"][suffix] = p

    for mode in ["soft", "hard", "weighted"]:
        key = f"voting_{mode}"
        for suffix in ["activity_code", "trial_has_fall", "is_fall"]:
            p = MODELS_DIR / f"xgboost_voting_{mode}_{suffix}.pkl"
            if p.exists():
                models[key][suffix] = p

    missing = []
    for k in ENSEMBLE_ORDER:
        for suffix in ["activity_code", "trial_has_fall", "is_fall"]:
            if suffix not in models[k]:
                missing.append(f"{k}:{suffix}")
    if missing:
        raise FileNotFoundError("Modèles d'ensemble manquants: " + ", ".join(missing))

    return models


def plot_bar_chart(metrics):
    groups = ["activity_code", "trial_has_fall", "is_fall"]
    group_labels = ["Activity", "TrialHasFall", "IsFall"]
    x = np.arange(len(group_labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_key in enumerate(ENSEMBLE_ORDER):
        vals = []
        for g in groups:
            if g == "activity_code":
                vals.append(metrics[model_key][g]["trial"] * 100)
            else:
                vals.append(metrics[model_key][g]["line"] * 100)

        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=ENSEMBLE_LABELS[i], color=COLOR_MAP[i])

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
    ax.set_title("Comparaison des modèles d'ensemble")
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

    for ax, key, name in zip(axes.flat, ENSEMBLE_ORDER, ENSEMBLE_LABELS):
        cm = cm_dict[key]
        denom = cm.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        norm = cm / denom * 100.0

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
                val = int(cm[i, j])
                pct = float(norm[i, j])
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
    feature_cols = build_feature_columns(train_df)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    _, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    _, y_test_trial, _, y_test_isfall = build_targets(train_df, test_df)
    n_classes = len(ac_to_id)

    model_paths = discover_ensemble_models()

    metrics = {k: {} for k in ENSEMBLE_ORDER}
    cms = {"activity_code": {}, "trial_has_fall": {}, "is_fall": {}}

    fnfp_activity = {k: (Counter(), Counter()) for k in ENSEMBLE_ORDER}
    fnfp_trial = {k: (Counter(), Counter()) for k in ENSEMBLE_ORDER}
    fnfp_isfall = {k: (Counter(), Counter()) for k in ENSEMBLE_ORDER}

    for model_key in ENSEMBLE_ORDER:
        paths = model_paths[model_key]

        model_ac = load_pickle(paths["activity_code"])
        y_pred_ac = np.asarray(model_ac.predict(X_test), dtype=int)
        row, trial, cm, trial_df_ac = eval_activity_preds(y_pred_ac, test_df, y_test_ac, ac_to_id, id_to_ac, n_classes)
        metrics[model_key]["activity_code"] = {"line": row, "trial": trial}
        cms["activity_code"][model_key] = cm
        fn, fp = fn_fp_from_multiclass_trials(trial_df_ac)
        fnfp_activity[model_key] = (fn, fp)

        model_trial = load_pickle(paths["trial_has_fall"])
        y_pred_trial = np.asarray(model_trial.predict(X_test), dtype=int)
        acc_trial, cm_trial, trial_df = eval_trial_preds(y_pred_trial, test_df, y_test_trial)
        metrics[model_key]["trial_has_fall"] = {"line": acc_trial}
        cms["trial_has_fall"][model_key] = cm_trial
        fn, fp = fn_fp_from_binary_df(trial_df, "true", "pred", "activity_code")
        fnfp_trial[model_key] = (fn, fp)

        model_isfall = load_pickle(paths["is_fall"])
        y_pred_isfall = np.asarray(model_isfall.predict(X_test), dtype=int)
        acc_isfall, cm_isfall, df_is = eval_isfall_preds(y_pred_isfall, test_df, y_test_isfall)
        metrics[model_key]["is_fall"] = {"line": acc_isfall}
        cms["is_fall"][model_key] = cm_isfall
        fn, fp = fn_fp_from_binary_df(df_is, "true", "pred", "activity_code")
        fnfp_isfall[model_key] = (fn, fp)

    plot_bar_chart(metrics)
    plot_confusions(cms["activity_code"], list(id_to_ac.values()), "Activity Code (Ensembles)")
    plot_confusions(cms["trial_has_fall"], ["Sans chute", "Chute"], "Trial Has Fall (Ensembles)")
    plot_confusions(cms["is_fall"], ["Non chute", "Chute"], "Is Fall (Ensembles)")

    plot_grouped_fn_fp("Erreurs par activité — Activity Code", fnfp_activity)
    plot_grouped_fn_fp("Erreurs par activité — Trial Has Fall", fnfp_trial)
    plot_grouped_fn_fp("Erreurs par activité — Is Fall", fnfp_isfall)


if __name__ == "__main__":
    main()
