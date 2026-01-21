import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import pickle
from pathlib import Path
from collections import Counter
from tqdm import tqdm

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")

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

N_TRIALS_DEFAULT = 200
N_TRIALS_WEIGHTED = 200
MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
THRESHOLD = 0.5

MODEL_DEFAULT_OUT = MODELS_DIR / "xgboost_hyper.pkl"
MODEL_WEIGHTED_OUT = MODELS_DIR / "xgboost_hyper_weighted.pkl"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def round_weight_to_half_tens(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").astype(np.float32)
    return (np.round(v / 5.0) * 5.0).astype(np.float32)


def gpu_available():
    try:
        d = xgb.DMatrix(np.array([[0.1, 0.2]], dtype=np.float32), label=[0])
        xgb.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return True
    except Exception:
        return False


def get_device_params():
    if gpu_available():
        print("✅ GPU XGBoost détecté et utilisé")
        return {"tree_method": "hist", "device": "cuda"}, "GPU"
    print("⚠️ GPU indisponible → fallback CPU")
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


def trial_level_confusion_from_proba(proba, test_df, y_test):
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


def score_fp0_min_fn(cm):
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    return float(fp * 1_000_000 + fn), fp, fn


def tune_hyperparams_default():
    train_df, test_df = load_data()

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["trial_has_fall"].astype(int).to_numpy()
    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df["trial_has_fall"].astype(int).to_numpy()

    device_params, backend = get_device_params()

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dtest = xgb.DMatrix(X_test, feature_names=FEATURES)

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            **device_params,
        }

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        best_iter = int(booster.best_iteration + 1)
        proba = booster.predict(dtest, iteration_range=(0, best_iter))
        cm = trial_level_confusion_from_proba(proba, test_df, y_test)
        score, fp, fn = score_fp0_min_fn(cm)

        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("fp", fp)
        trial.set_user_attr("fn", fn)
        return score

    study = optuna.create_study(direction="minimize", study_name="xgb_default")

    pbar = tqdm(total=N_TRIALS_DEFAULT, desc=f"Optuna default [{backend}]", dynamic_ncols=True)
    for _ in range(N_TRIALS_DEFAULT):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        pbar.update(1)
    pbar.close()

    best_params = dict(study.best_params)
    best_iter = int(study.best_trial.user_attrs["best_iteration"])

    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        **device_params,
        **best_params,
    }

    final = xgb.train(final_params, dtrain, num_boost_round=best_iter)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DEFAULT_OUT, "wb") as f:
        pickle.dump(final, f)

    proba = final.predict(dtest)
    cm = trial_level_confusion_from_proba(proba, test_df, y_test)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    print(f"\n✅ saved -> {MODEL_DEFAULT_OUT.resolve()}")
    print(f"default [{backend}] best_iter={best_iter} | TN={tn} FP={fp} FN={fn} TP={tp} | acc={acc:.6f}")


def tune_hyperparams_weighted():
    train_df, test_df = load_data()

    X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
    y_train = train_df["trial_has_fall"].astype(int).to_numpy()
    X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
    y_test = test_df["trial_has_fall"].astype(int).to_numpy()

    device_params, backend = get_device_params()
    base_spw = float(compute_scale_pos_weight(y_train))

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
    dtest = xgb.DMatrix(X_test, feature_names=FEATURES)

    def objective(trial):
        spw_mult = trial.suggest_float("spw_mult", 0.5, 2.0)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": float(base_spw * spw_mult),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 25.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            **device_params,
        }

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=MAX_BOOST_ROUNDS,
            evals=[(dtrain, "train")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        best_iter = int(booster.best_iteration + 1)
        proba = booster.predict(dtest, iteration_range=(0, best_iter))
        cm = trial_level_confusion_from_proba(proba, test_df, y_test)
        score, fp, fn = score_fp0_min_fn(cm)

        trial.set_user_attr("best_iteration", best_iter)
        trial.set_user_attr("fp", fp)
        trial.set_user_attr("fn", fn)
        return score

    study = optuna.create_study(direction="minimize", study_name="xgb_weighted")

    pbar = tqdm(total=N_TRIALS_WEIGHTED, desc=f"Optuna pondéré [{backend}]", dynamic_ncols=True)
    for _ in range(N_TRIALS_WEIGHTED):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        pbar.update(1)
    pbar.close()

    best_params = dict(study.best_params)
    best_iter = int(study.best_trial.user_attrs["best_iteration"])

    spw_mult = float(best_params.pop("spw_mult"))
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "scale_pos_weight": float(base_spw * spw_mult),
        **device_params,
        **best_params,
    }

    final = xgb.train(final_params, dtrain, num_boost_round=best_iter)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_WEIGHTED_OUT, "wb") as f:
        pickle.dump(final, f)

    proba = final.predict(dtest)
    cm = trial_level_confusion_from_proba(proba, test_df, y_test)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    print(f"\n✅ saved -> {MODEL_WEIGHTED_OUT.resolve()}")
    print(f"pondéré [{backend}] best_iter={best_iter} | TN={tn} FP={fp} FN={fn} TP={tp} | acc={acc:.6f}")
    print(f"base_spw={base_spw:.6f} | chosen scale_pos_weight={float(final_params['scale_pos_weight']):.6f}")


def main():
    tune_hyperparams_default()
    tune_hyperparams_weighted()


if __name__ == "__main__":
    main()
