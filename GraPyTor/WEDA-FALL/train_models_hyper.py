import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from tqdm import tqdm
import pickle
import optuna
from sklearn.model_selection import train_test_split

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

N_TRIALS_ACTIVITY = 50
N_TRIALS_TRIAL_HAS_FALL = 50
N_TRIALS_IS_FALL = 50
MAX_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 50

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude_cols = {"user_id", "trial_id", "activity_code", "trial_has_fall", "is_fall"}
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in "fc"]
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


def tune_activity_model(X, y, n_classes, n_trials):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    study = optuna.create_study(direction="minimize")

    outer_pbar = tqdm(
        total=n_trials,
        desc="Optuna activity_code",
        position=0,
        leave=False,
        dynamic_ncols=True
    )

    for _ in range(n_trials):
        inner_pbar = tqdm(
            total=MAX_BOOST_ROUNDS,
            desc="activity_code boosting",
            position=1,
            leave=False,
            dynamic_ncols=True
        )

        cb = TqdmXgbCallback(inner_pbar, MAX_BOOST_ROUNDS)

        def objective(trial):
            params = {
                "objective": "multi:softprob",
                "num_class": n_classes,
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            }

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=MAX_BOOST_ROUNDS,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                callbacks=[cb]
            )

            trial.set_user_attr("best_iteration", booster.best_iteration)
            return float(booster.best_score)

        study.optimize(objective, n_trials=1)
        inner_pbar.close()
        outer_pbar.update(1)

    outer_pbar.close()

    best_params = study.best_params
    best_params.update(
        {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "tree_method": "hist"
        }
    )
    best_iter = study.best_trial.user_attrs.get("best_iteration", 500)

    dtrain_full = xgb.DMatrix(X, label=y)

    final_pbar = tqdm(
        total=best_iter,
        desc="activity_code final",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    final_cb = TqdmXgbCallback(final_pbar, best_iter)

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain_full,
        num_boost_round=best_iter,
        callbacks=[final_cb]
    )

    final_pbar.close()
    return booster


def tune_binary_model(X, y, n_trials, name):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    study = optuna.create_study(direction="minimize")

    outer_pbar = tqdm(
        total=n_trials,
        desc=f"Optuna {name}",
        position=0,
        leave=False,
        dynamic_ncols=True
    )

    for _ in range(n_trials):
        inner_pbar = tqdm(
            total=MAX_BOOST_ROUNDS,
            desc=f"{name} boosting",
            position=1,
            leave=False,
            dynamic_ncols=True
        )

        cb = TqdmXgbCallback(inner_pbar, MAX_BOOST_ROUNDS)

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
            }

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=MAX_BOOST_ROUNDS,
                evals=[(dtrain, "train"), (dvalid, "valid")],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
                callbacks=[cb]
            )
            trial.set_user_attr("best_iteration", booster.best_iteration)
            return float(booster.best_score)

        study.optimize(objective, n_trials=1)
        inner_pbar.close()
        outer_pbar.update(1)

    outer_pbar.close()

    best_params = study.best_params
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist"
        }
    )
    best_iter = study.best_trial.user_attrs.get("best_iteration", 500)

    dtrain_full = xgb.DMatrix(X, label=y)

    final_pbar = tqdm(
        total=best_iter,
        desc=f"{name} final",
        position=0,
        leave=True,
        dynamic_ncols=True
    )

    final_cb = TqdmXgbCallback(final_pbar, best_iter)

    booster = xgb.train(
        params=best_params,
        dtrain=dtrain_full,
        num_boost_round=best_iter,
        callbacks=[final_cb]
    )

    final_pbar.close()
    return booster


def save_model_hyper(name, booster):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"xgboost_hyper_{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(booster, f)
    print(f"Modèle sauvegardé: {path.resolve()}")


def main():
    train_df, test_df = load_data()
    feature_cols = build_feature_columns(train_df)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)

    y_train_ac, _, ac_to_id, _ = encode_activity(train_df, test_df)
    y_train_trial, _, y_train_isfall, _ = build_targets(train_df, test_df)

    n_classes = len(ac_to_id)

    booster_ac = tune_activity_model(X_train, y_train_ac, n_classes, N_TRIALS_ACTIVITY)
    save_model_hyper("activity_code", booster_ac)

    booster_trial = tune_binary_model(X_train, y_train_trial, N_TRIALS_TRIAL_HAS_FALL, "trial_has_fall")
    save_model_hyper("trial_has_fall", booster_trial)

    booster_isfall = tune_binary_model(X_train, y_train_isfall, N_TRIALS_IS_FALL, "is_fall")
    save_model_hyper("is_fall", booster_isfall)


if __name__ == "__main__":
    main()
