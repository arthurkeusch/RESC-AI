import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from tqdm import tqdm
import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

N_TRIALS_ACTIVITY = 50
N_TRIALS_TRIAL_HAS_FALL = 50
N_TRIALS_IS_FALL = 50
MAX_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 50

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_xgb_device_params():
    import numpy as _np
    import xgboost as _xgb
    try:
        d = _xgb.DMatrix(_np.array([[0.1, 0.2]]), label=[0])
        _xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        print("✅ GPU XGBoost détecté et utilisé (hyperparamètres inclus)")
        return {"tree_method": "hist", "device": "cuda"}
    except Exception as e:
        print("⚠️ GPU indisponible → fallback CPU :", e)
        return {"tree_method": "hist"}


DEVICE_PARAMS = get_xgb_device_params()


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude_cols = {"user_id", "trial_id", "activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in "fc"]


def encode_activity(train_df):
    ac_train = sorted(train_df["activity_code"].unique())
    ac_to_id = {ac: i for i, ac in enumerate(ac_train)}
    y = train_df["activity_code"].map(ac_to_id).astype(int).to_numpy()
    return y, ac_to_id


def build_targets(train_df):
    return (
        train_df["trial_has_fall"].astype(int).to_numpy(),
        train_df["is_fall"].astype(int).to_numpy()
    )


def compute_scale_pos_weight(y):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return 1.0 if pos == 0 else neg / pos


class TqdmXgbCallback(xgb.callback.TrainingCallback):
    def __init__(self, pbar, total):
        self.pbar = pbar
        self.total = total

    def before_training(self, model):
        self.pbar.reset(total=self.total)
        self.pbar.n = 0
        self.pbar.refresh()
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        if self.pbar.n < self.total:
            self.pbar.n = self.total
        self.pbar.refresh()
        return model


def tune_model_common(objective_builder, name, X, y, n_trials):
    study = optuna.create_study(direction="minimize")
    outer = tqdm(total=n_trials, desc=f"Optuna {name}", dynamic_ncols=True, leave=False)

    for _ in range(n_trials):
        inner = tqdm(total=MAX_BOOST_ROUNDS, desc=f"{name} boosting", dynamic_ncols=True, leave=False)
        cb = TqdmXgbCallback(inner, MAX_BOOST_ROUNDS)

        def objective(trial):
            score, best_iter = objective_builder(trial, cb)
            trial.set_user_attr("best_iteration", best_iter)
            return score

        study.optimize(objective, n_trials=1)
        inner.close()
        outer.update(1)

    outer.close()
    return study


def train_final_model(best_params, best_iter, X, y, name):
    dtrain = xgb.DMatrix(X, label=y)
    final = tqdm(total=best_iter, desc=f"{name} final", dynamic_ncols=True)
    booster = xgb.train(best_params, dtrain, num_boost_round=best_iter, callbacks=[TqdmXgbCallback(final, best_iter)])
    final.close()
    return booster


def evaluate_optuna_score(model, X, y, multiclass):
    d = xgb.DMatrix(X, label=y)
    p = model.predict(d)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(log_loss(y, p))


def save_if_better(name, booster, X, y, weighted, multiclass):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "xgboost_hyper_pond_" if weighted else "xgboost_hyper_"
    model_path = MODELS_DIR / f"{prefix}{name}.pkl"

    new_score = evaluate_optuna_score(booster, X, y, multiclass)
    print(f"🧪 {name} : logloss nouveau modèle = {new_score:.6f}")

    if not model_path.exists():
        with open(model_path, "wb") as f:
            pickle.dump(booster, f)
        print("✅ Aucun ancien modèle → sauvegarde effectuée")
        return

    with open(model_path, "rb") as f:
        old_model = pickle.load(f)

    old_score = evaluate_optuna_score(old_model, X, y, multiclass)
    print(f"📁 {name} : logloss ancien modèle = {old_score:.6f}")

    if new_score < old_score:
        with open(model_path, "wb") as f:
            pickle.dump(booster, f)
        print("✅ Nouveau modèle meilleur → sauvegardé")
    else:
        print("❌ Ancien modèle meilleur ou égal → conservé")


def tune_activity_model(X, y, n_classes, n_trials, weighted=False, scale_pos_weight=None):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective_builder(trial, cb):
        params = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            **DEVICE_PARAMS
        }

        booster = xgb.train(
            params, dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            callbacks=[cb]
        )

        return float(booster.best_score), booster.best_iteration

    study = tune_model_common(objective_builder, "activity_code", X, y, n_trials)

    best_params = study.best_params
    best_params.update({
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        **DEVICE_PARAMS
    })

    best_iter = study.best_trial.user_attrs["best_iteration"]
    booster = train_final_model(best_params, best_iter, X, y, "activity_code")

    return booster


def tune_binary_model(X, y, n_trials, name, weighted=False, scale_pos_weight=None):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective_builder(trial, cb):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            **DEVICE_PARAMS
        }

        if weighted:
            params["scale_pos_weight"] = scale_pos_weight

        booster = xgb.train(
            params, dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
            callbacks=[cb]
        )

        return float(booster.best_score), booster.best_iteration

    study = tune_model_common(objective_builder, name, X, y, n_trials)

    best_params = study.best_params
    best_params.update({"objective": "binary:logistic", "eval_metric": "logloss", **DEVICE_PARAMS})

    if weighted:
        best_params["scale_pos_weight"] = scale_pos_weight

    best_iter = study.best_trial.user_attrs["best_iteration"]
    booster = train_final_model(best_params, best_iter, X, y, name)

    return booster


def main():
    train_df, _ = load_data()
    X = train_df[build_feature_columns(train_df)].to_numpy(dtype=np.float32)
    y_ac, ac_to_id = encode_activity(train_df)
    y_trial, y_isfall = build_targets(train_df)

    scale_trial = compute_scale_pos_weight(y_trial)
    scale_isfall = compute_scale_pos_weight(y_isfall)

    print("\n===== SANS PONDÉRATION =====\n")

    booster = tune_activity_model(X, y_ac, len(ac_to_id), N_TRIALS_ACTIVITY, False)
    save_if_better("activity_code", booster, X, y_ac, False, True)

    booster = tune_binary_model(X, y_trial, N_TRIALS_TRIAL_HAS_FALL, "trial_has_fall", False)
    save_if_better("trial_has_fall", booster, X, y_trial, False, False)

    booster = tune_binary_model(X, y_isfall, N_TRIALS_IS_FALL, "is_fall", False)
    save_if_better("is_fall", booster, X, y_isfall, False, False)

    print("\n===== AVEC PONDÉRATION =====\n")

    booster = tune_activity_model(X, y_ac, len(ac_to_id), N_TRIALS_ACTIVITY, True)
    save_if_better("activity_code", booster, X, y_ac, True, True)

    booster = tune_binary_model(X, y_trial, N_TRIALS_TRIAL_HAS_FALL, "trial_has_fall", True, scale_trial)
    save_if_better("trial_has_fall", booster, X, y_trial, True, False)

    booster = tune_binary_model(X, y_isfall, N_TRIALS_IS_FALL, "is_fall", True, scale_isfall)
    save_if_better("is_fall", booster, X, y_isfall, True, False)

    print("✅ Meilleurs modèles conservés (jamais écrasés inutilement).")


if __name__ == "__main__":
    main()
