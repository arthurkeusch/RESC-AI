import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

TASKS = ["activity_code", "trial_has_fall", "is_fall"]
VARIANTS = ["standard", "pond", "hyper", "hyper_pond"]


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude_targets = {"activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude_targets and df[c].dtype.kind in "fc"]


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

    def _predict_proba_activity_weighted(self, dm: xgb.DMatrix):
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

    def _predict_proba_binary_weighted(self, dm: xgb.DMatrix):
        psum = 0.0
        wsum = 0.0
        for k, b in self.boosters.items():
            p = b.predict(dm)
            w = self._w(k)
            psum += p * w
            wsum += w
        return psum / max(wsum, 1e-12)

    def _hard_vote_multiclass(self, dm: xgb.DMatrix):
        preds = []
        for _, b in self.boosters.items():
            p = b.predict(dm)
            preds.append(np.argmax(p, axis=1).astype(int))
        preds = np.stack(preds, axis=0)
        return np.apply_along_axis(lambda a: np.bincount(a).argmax(), 0, preds).astype(int)

    def _hard_vote_binary(self, dm: xgb.DMatrix):
        preds = []
        for _, b in self.boosters.items():
            p = b.predict(dm)
            preds.append((p >= self.threshold).astype(int))
        preds = np.stack(preds, axis=0)
        return (np.sum(preds, axis=0) >= (preds.shape[0] / 2)).astype(int)

    def predict(self, X: np.ndarray):
        if self.mode not in {"soft", "hard", "weighted"}:
            raise ValueError("mode doit être 'soft', 'hard' ou 'weighted'")

        dm = xgb.DMatrix(X)

        if self.task == "activity_code":
            if self.mode == "hard":
                return self._hard_vote_multiclass(dm)
            probs = self._predict_proba_activity_weighted(dm)
            return np.argmax(probs, axis=1).astype(int)

        if self.task in {"trial_has_fall", "is_fall"}:
            if self.mode == "hard":
                return self._hard_vote_binary(dm)
            p = self._predict_proba_binary_weighted(dm)
            return (p >= self.threshold).astype(int)

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

    def _meta_features_binary(self, dm: xgb.DMatrix):
        return np.stack([self.boosters[k].predict(dm) for k in VARIANTS], axis=1).astype(np.float32)

    def _meta_features_multiclass(self, dm: xgb.DMatrix):
        feats = []
        for k in VARIANTS:
            feats.append(self.boosters[k].predict(dm))
        return np.concatenate(feats, axis=1).astype(np.float32)

    def predict_proba(self, X):
        dm = xgb.DMatrix(X)
        if self.task == "activity_code":
            F = self._meta_features_multiclass(dm).astype(np.float64)
            logits = F @ self.W + self.b
            return self._softmax(logits)
        F = self._meta_features_binary(dm).astype(np.float64)
        p = self._sigmoid(F @ self.w + self.b)
        return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        if self.task == "activity_code":
            p = self.predict_proba(X)
            return np.argmax(p, axis=1).astype(int)
        p = self.predict_proba(X)[:, 1]
        return (p >= self.threshold).astype(int)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def task_from_filename(stem: str):
    for t in TASKS:
        if stem.endswith("_" + t):
            return t
    return None


def discover_models():
    out = {t: [] for t in TASKS}
    for p in sorted(MODELS_DIR.glob("*.pkl")):
        t = task_from_filename(p.stem)
        if t is None:
            continue
        out[t].append(p)
    missing = [t for t in TASKS if not out[t]]
    if missing:
        raise FileNotFoundError("Aucun modèle trouvé pour: " + ", ".join(missing))
    return out


def predict_for_task(model, task: str, X: np.ndarray):
    if isinstance(model, xgb.Booster):
        dm = xgb.DMatrix(X)
        p = model.predict(dm)
        if task == "activity_code":
            return np.argmax(p, axis=1).astype(int)
        return (p >= 0.5).astype(int)

    if hasattr(model, "get_booster"):
        b = model.get_booster()
        return predict_for_task(b, task, X)

    if hasattr(model, "predict"):
        y = model.predict(X)
        y = np.asarray(y)
        if task == "activity_code":
            if y.ndim == 2:
                return np.argmax(y, axis=1).astype(int)
            return y.astype(int)
        return y.astype(int)

    raise TypeError(f"Type de modèle non supporté: {type(model)}")


def bench_predict_cpu_ms_per_row(model, task: str, X: np.ndarray, repeats: int = 7, warmup: int = 2):
    n = int(len(X))
    if n == 0:
        return float("nan")

    for _ in range(warmup):
        _ = predict_for_task(model, task, X)

    times = []
    for _ in range(repeats):
        t0 = time.process_time_ns()
        _ = predict_for_task(model, task, X)
        t1 = time.process_time_ns()
        times.append((t1 - t0) / 1e6)

    times.sort()
    cpu_ms_total = times[len(times) // 2]
    return cpu_ms_total / n


def pretty_name(path: Path):
    s = path.stem
    s = s.replace("xgboost_", "")
    s = s.replace("_activity_code", "")
    s = s.replace("_trial_has_fall", "")
    s = s.replace("_is_fall", "")
    return s


def plot_bar(title: str, names: list[str], values: list[float]):
    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(names) + 1)))
    bars = ax.barh(names, values)
    ax.set_title(title)
    ax.set_xlabel("CPU time moyen par prédiction (ms / ligne)")

    vmax = max(values) if values else 1.0
    pad = vmax * 0.01
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2, f"{v:.6f}", va="center")

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def main():
    train_df, test_df = load_data()
    feature_cols = build_feature_columns(train_df)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    model_paths = discover_models()

    for task in TASKS:
        names = []
        cpu_ms_per_row = []

        for p in tqdm(model_paths[task], desc=f"Benchmark {task}", unit="model"):
            model = load_pickle(p)
            ms = bench_predict_cpu_ms_per_row(model, task, X_test, repeats=7, warmup=2)
            names.append(pretty_name(p))
            cpu_ms_per_row.append(ms)

        plot_bar(f"Benchmark CPU par prédiction — {task}", names, cpu_ms_per_row)


if __name__ == "__main__":
    main()
