import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import xgboost as xgb
from tqdm import tqdm
import pickle

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")
MODELS_DIR = Path("models")


def get_xgb_device_params():
    try:
        test = xgb.DMatrix(np.array([[0.1, 0.2]]), label=[0])
        xgb.train(
            {"tree_method": "hist", "device": "cuda"},
            test,
            num_boost_round=1
        )
        print("✅ GPU XGBoost détecté et utilisé")
        return {"tree_method": "hist", "device": "cuda"}
    except Exception as e:
        print("⚠️ GPU non disponible, retour CPU")
        print("   Raison :", e)
        return {"tree_method": "hist"}


DEVICE_PARAMS = get_xgb_device_params()


class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total, desc, position=0):
        self.total = total
        self.desc = desc
        self.position = position
        self.pbar = None

    def before_training(self, model):
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            unit="iter",
            position=self.position,
            leave=True,
            dynamic_ncols=True
        )
        return model

    def after_iteration(self, model, epoch, evals_log):
        if self.pbar:
            self.pbar.update(1)
        return False

    def after_training(self, model):
        if self.pbar:
            self.pbar.close()
        return model


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


def compute_scale_pos_weight(y):
    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)
    if positives == 0:
        return 1.0
    return negatives / positives


def train_models(X_train, y_ac, y_trial, y_isfall, n_classes, scale_trial=None, scale_isfall=None, num_rounds=200):
    weighted = scale_trial is not None and scale_isfall is not None
    tag = "POND" if weighted else "STANDARD"

    if weighted:
        print(f"[{tag}] scale_pos_weight trial_has_fall = {scale_trial:.2f}")
        print(f"[{tag}] scale_pos_weight is_fall = {scale_isfall:.2f}\n")

    print(f"=== [{tag}] Entraînement modèle 1/3 : activity_code ===")
    params_ac = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        **DEVICE_PARAMS
    }

    dtrain_ac = xgb.DMatrix(X_train, label=y_ac)
    booster_ac = xgb.train(
        params=params_ac,
        dtrain=dtrain_ac,
        num_boost_round=num_rounds,
        evals=[(dtrain_ac, "train")],
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, f"{tag} activity_code", position=0)]
    )

    print(f"=== [{tag}] Entraînement modèle 2/3 : trial_has_fall ===")
    params_trial = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        **DEVICE_PARAMS
    }

    if weighted:
        params_trial["scale_pos_weight"] = scale_trial

    dtrain_trial = xgb.DMatrix(X_train, label=y_trial)
    booster_trial = xgb.train(
        params=params_trial,
        dtrain=dtrain_trial,
        num_boost_round=num_rounds,
        evals=[(dtrain_trial, "train")],
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, f"{tag} trial_has_fall", position=0)]
    )

    print(f"=== [{tag}] Entraînement modèle 3/3 : is_fall ===")
    params_isfall = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        **DEVICE_PARAMS
    }

    if weighted:
        params_isfall["scale_pos_weight"] = scale_isfall

    dtrain_isfall = xgb.DMatrix(X_train, label=y_isfall)
    booster_isfall = xgb.train(
        params=params_isfall,
        dtrain=dtrain_isfall,
        num_boost_round=num_rounds,
        evals=[(dtrain_isfall, "train")],
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, f"{tag} is_fall", position=0)]
    )

    return booster_ac, booster_trial, booster_isfall, weighted


def save_models(booster_ac, booster_trial, booster_isfall, weighted):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "xgboost_pond_" if weighted else "xgboost_"

    with open(MODELS_DIR / f"{prefix}activity_code.pkl", "wb") as f:
        pickle.dump(booster_ac, f)
    with open(MODELS_DIR / f"{prefix}trial_has_fall.pkl", "wb") as f:
        pickle.dump(booster_trial, f)
    with open(MODELS_DIR / f"{prefix}is_fall.pkl", "wb") as f:
        pickle.dump(booster_isfall, f)

    tag = "POND" if weighted else "STANDARD"
    print(f"[{tag}] Modèles sauvegardés dans {MODELS_DIR.resolve()}\n")


def main():
    train_df, test_df = load_data()
    feature_cols = build_feature_columns(train_df)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)

    y_train_ac, _, ac_to_id, _ = encode_activity(train_df, test_df)
    y_train_trial, _, y_train_isfall, _ = build_targets(train_df, test_df)

    scale_trial = compute_scale_pos_weight(y_train_trial)
    scale_isfall = compute_scale_pos_weight(y_train_isfall)

    print("\n===== ENTRAÎNEMENT SANS PONDÉRATION =====\n")
    booster_ac, booster_trial, booster_isfall, weighted = train_models(
        X_train,
        y_train_ac,
        y_train_trial,
        y_train_isfall,
        len(ac_to_id)
    )
    save_models(booster_ac, booster_trial, booster_isfall, weighted)

    print("\n===== ENTRAÎNEMENT AVEC PONDÉRATION =====\n")
    booster_ac, booster_trial, booster_isfall, weighted = train_models(
        X_train,
        y_train_ac,
        y_train_trial,
        y_train_isfall,
        len(ac_to_id),
        scale_trial,
        scale_isfall
    )
    save_models(booster_ac, booster_trial, booster_isfall, weighted)

    print("✅ Tous les modèles ont été générés (avec GPU si dispo, sinon CPU).")


if __name__ == "__main__":
    main()
