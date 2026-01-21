import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import pickle
import optuna
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

VARIANTS = ["standard", "pond", "hyper", "hyper_pond"]

NB_TRIALS = 200


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude_targets = {"activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude_targets and df[c].dtype.kind in "fc"]


def encode_activity(train_df):
    ac_train = sorted(train_df["activity_code"].unique())
    ac_to_id = {ac: i for i, ac in enumerate(ac_train)}
    id_to_ac = {i: ac for ac, i in ac_to_id.items()}
    return ac_to_id, id_to_ac


def load_xgboost_booster(path: Path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    if isinstance(model, xgb.Booster):
        return model
    return model.get_booster()


def load_4_boosters(suffix: str):
    paths = {
        "standard": MODELS_DIR / f"xgboost_{suffix}.pkl",
        "pond": MODELS_DIR / f"xgboost_pond_{suffix}.pkl",
        "hyper": MODELS_DIR / f"xgboost_hyper_{suffix}.pkl",
        "hyper_pond": MODELS_DIR / f"xgboost_hyper_pond_{suffix}.pkl",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Modèles manquants pour '{suffix}': {missing}")
    return {k: load_xgboost_booster(p) for k, p in paths.items()}


def normalize_weights(w: dict):
    s = float(sum(w.values()))
    if s <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: float(v) / s for k, v in w.items()}


def build_precomputed_probs_binary(boosters: dict, X: np.ndarray):
    dm = xgb.DMatrix(X)
    return {k: boosters[k].predict(dm) for k in ["standard", "pond", "hyper", "hyper_pond"]}


def silence_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("optuna.study").setLevel(logging.WARNING)
    logging.getLogger("optuna.trial").setLevel(logging.WARNING)


def tune_weighted_voting_binary(train_df: pd.DataFrame, feature_cols: list[str], target_col: str, n_trials: int = 200, seed: int = 42):
    idx_train, idx_valid = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=seed,
        stratify=train_df[target_col].astype(int),
    )
    valid_df = train_df.iloc[idx_valid].reset_index(drop=True)
    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_valid = valid_df[target_col].astype(int).to_numpy()

    boosters = load_4_boosters(target_col)
    probs = build_precomputed_probs_binary(boosters, X_valid)

    best_score = -1.0
    best_w = None
    best_thr = 0.5

    def objective(trial):
        nonlocal best_score, best_w, best_thr

        w = {
            "standard": trial.suggest_float("w_standard", 0.05, 3.0, log=True),
            "pond": trial.suggest_float("w_pond", 0.05, 3.0, log=True),
            "hyper": trial.suggest_float("w_hyper", 0.05, 3.0, log=True),
            "hyper_pond": trial.suggest_float("w_hyper_pond", 0.05, 3.0, log=True),
        }
        w = normalize_weights(w)
        thr = trial.suggest_float("threshold", 0.05, 0.95)

        p = (
            probs["standard"] * w["standard"]
            + probs["pond"] * w["pond"]
            + probs["hyper"] * w["hyper"]
            + probs["hyper_pond"] * w["hyper_pond"]
        )

        y_pred = (p >= thr).astype(int)
        score = float((y_pred == y_valid).mean())

        if score > best_score:
            best_score = score
            best_w = w
            best_thr = thr

        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    pbar = tqdm(total=n_trials, desc=f"Optuna weights ({target_col})", unit="trial")

    def cb(study, trial):
        pbar.update(1)

    study.optimize(objective, n_trials=n_trials, callbacks=[cb], show_progress_bar=False)
    pbar.close()

    return boosters, best_w, float(best_thr)


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
                dm = xgb.DMatrix(X)
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
            p = self.boosters[k].predict(dm)
            feats.append(p)
        return np.concatenate(feats, axis=1).astype(np.float32)

    def predict_proba(self, X):
        if self.task == "activity_code":
            F = self._meta_features_multiclass(X)
            logits = F @ self.W + self.b
            return self._softmax(logits)
        else:
            F = self._meta_features_binary(X)
            p = self._sigmoid(F @ self.w + self.b)
            return np.vstack([1.0 - p, p]).T

    def predict(self, X):
        if self.task == "activity_code":
            p = self.predict_proba(X)
            return np.argmax(p, axis=1).astype(int)
        else:
            p = self.predict_proba(X)[:, 1]
            return (p >= self.threshold).astype(int)


def fit_binary_meta_optuna(P, y, n_trials=200, seed=42, desc="Stacking meta (binary)"):
    silence_optuna()
    best = {"score": -1.0, "w": None, "b": 0.0, "thr": 0.5}

    def objective(trial):
        w = np.array(
            [
                trial.suggest_float("w_standard", -6.0, 6.0),
                trial.suggest_float("w_pond", -6.0, 6.0),
                trial.suggest_float("w_hyper", -6.0, 6.0),
                trial.suggest_float("w_hyper_pond", -6.0, 6.0),
            ],
            dtype=np.float64,
        )
        b = float(trial.suggest_float("bias", -6.0, 6.0))
        thr = float(trial.suggest_float("threshold", 0.05, 0.95))
        z = P @ w + b
        z = np.clip(z, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        y_pred = (p >= thr).astype(int)
        score = float((y_pred == y).mean())
        if score > best["score"]:
            best["score"] = score
            best["w"] = w
            best["b"] = b
            best["thr"] = thr
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    pbar = tqdm(total=n_trials, desc=desc, unit="trial")

    def cb(study, trial):
        pbar.update(1)

    study.optimize(objective, n_trials=n_trials, callbacks=[cb], show_progress_bar=False)
    pbar.close()

    return best["w"], best["b"], best["thr"]


def fit_multiclass_meta_ridge(P, y, n_classes, reg=1e-3):
    Y = np.eye(n_classes, dtype=np.float64)[y]
    A = P.T @ P + reg * np.eye(P.shape[1], dtype=np.float64)
    B = P.T @ Y
    W = np.linalg.solve(A, B)
    b = np.zeros((n_classes,), dtype=np.float64)
    return W, b


def build_activity_voting(train_df, mode: str):
    ac_to_id, id_to_ac = encode_activity(train_df)
    boosters = load_4_boosters("activity_code")
    meta = {"ac_to_id": ac_to_id, "id_to_ac": id_to_ac, "n_classes": len(ac_to_id)}
    return VotingEnsemble(task="activity_code", boosters=boosters, mode=mode, threshold=0.5, weights=None, meta=meta)


def build_trial_voting(train_df, mode: str, feature_cols: list[str]):
    if mode in {"soft", "hard"}:
        boosters = load_4_boosters("trial_has_fall")
        return VotingEnsemble(task="trial_has_fall", boosters=boosters, mode=mode, threshold=0.5, weights=None, meta={})
    boosters, best_w, best_thr = tune_weighted_voting_binary(train_df, feature_cols, "trial_has_fall", n_trials=NB_TRIALS, seed=42)
    return VotingEnsemble(task="trial_has_fall", boosters=boosters, mode="weighted", threshold=best_thr, weights=best_w, meta={})


def build_isfall_voting(train_df, mode: str, feature_cols: list[str]):
    if mode in {"soft", "hard"}:
        boosters = load_4_boosters("is_fall")
        return VotingEnsemble(task="is_fall", boosters=boosters, mode=mode, threshold=0.5, weights=None, meta={})
    boosters, best_w, best_thr = tune_weighted_voting_binary(train_df, feature_cols, "is_fall", n_trials=NB_TRIALS, seed=42)
    return VotingEnsemble(task="is_fall", boosters=boosters, mode="weighted", threshold=best_thr, weights=best_w, meta={})


def build_stacking_binary(train_df, feature_cols: list[str], target_col: str):
    idx_train, idx_valid = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=42,
        stratify=train_df[target_col].astype(int),
    )
    valid_df = train_df.iloc[idx_valid].reset_index(drop=True)
    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_valid = valid_df[target_col].astype(int).to_numpy()

    boosters = load_4_boosters(target_col)
    dm = xgb.DMatrix(X_valid)
    P = np.stack([boosters[k].predict(dm) for k in VARIANTS], axis=1).astype(np.float64)

    w, b, thr = fit_binary_meta_optuna(P, y_valid, n_trials=NB_TRIALS, seed=42, desc=f"Stacking meta ({target_col})")

    model = StackingEnsemble(task=target_col, boosters=boosters, meta={})
    model.w = w.astype(np.float64)
    model.b = float(b)
    model.threshold = float(thr)
    return model


def build_stacking_activity(train_df, feature_cols: list[str]):
    ac_to_id, id_to_ac = encode_activity(train_df)
    idx_train, idx_valid = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=42,
        stratify=train_df["trial_has_fall"].astype(int),
    )
    valid_df = train_df.iloc[idx_valid].reset_index(drop=True)
    X_valid = valid_df[feature_cols].to_numpy(dtype=np.float32)
    y_valid = valid_df["activity_code"].map(ac_to_id).fillna(-1).astype(int).to_numpy()

    mask = y_valid >= 0
    X_valid = X_valid[mask]
    y_valid = y_valid[mask]

    boosters = load_4_boosters("activity_code")
    dm = xgb.DMatrix(X_valid)
    probs_list = [boosters[k].predict(dm) for k in VARIANTS]
    P = np.concatenate(probs_list, axis=1).astype(np.float64)

    W, b = fit_multiclass_meta_ridge(P, y_valid.astype(int), len(ac_to_id), reg=1e-3)

    model = StackingEnsemble(task="activity_code", boosters=boosters, meta={"ac_to_id": ac_to_id, "id_to_ac": id_to_ac, "n_classes": len(ac_to_id)})
    model.W = W.astype(np.float64)
    model.b = b.astype(np.float64)
    model.n_classes = len(ac_to_id)
    return model


def save_model(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main():
    silence_optuna()

    train_df, _ = load_data()
    feature_cols = build_feature_columns(train_df)

    for mode in ["soft", "hard", "weighted"]:
        voting_activity = build_activity_voting(train_df, mode)
        voting_trial = build_trial_voting(train_df, mode, feature_cols)
        voting_isfall = build_isfall_voting(train_df, mode, feature_cols)

        save_model(voting_activity, MODELS_DIR / f"xgboost_voting_{mode}_activity_code.pkl")
        save_model(voting_trial, MODELS_DIR / f"xgboost_voting_{mode}_trial_has_fall.pkl")
        save_model(voting_isfall, MODELS_DIR / f"xgboost_voting_{mode}_is_fall.pkl")

    stacking_activity = build_stacking_activity(train_df, feature_cols)
    stacking_trial = build_stacking_binary(train_df, feature_cols, "trial_has_fall")
    stacking_isfall = build_stacking_binary(train_df, feature_cols, "is_fall")

    save_model(stacking_activity, MODELS_DIR / "xgboost_stacking_activity_code.pkl")
    save_model(stacking_trial, MODELS_DIR / "xgboost_stacking_trial_has_fall.pkl")
    save_model(stacking_isfall, MODELS_DIR / "xgboost_stacking_is_fall.pkl")


if __name__ == "__main__":
    main()
