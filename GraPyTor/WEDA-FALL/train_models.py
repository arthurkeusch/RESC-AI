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

class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total, desc):
        self.total = total
        self.desc = desc
        self.pbar = None

    def before_training(self, model):
        self.pbar = tqdm(total=self.total, desc=self.desc, unit="iter")
        return model

    def after_iteration(self, model, epoch, evals_log):
        if self.pbar is not None:
            self.pbar.update(1)
        return False

    def after_training(self, model):
        if self.pbar is not None:
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

def train_models(X_train, y_ac, y_trial, y_isfall, n_classes, num_rounds=200):
    print("=== Entraînement modèle 1/3 : activity_code ===")
    params_ac = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    evals_result_ac = {}
    dtrain_ac = xgb.DMatrix(X_train, label=y_ac)
    booster_ac = xgb.train(
        params=params_ac,
        dtrain=dtrain_ac,
        num_boost_round=num_rounds,
        evals=[(dtrain_ac, "train")],
        evals_result=evals_result_ac,
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, "activity_code")]
    )
    final_ac = evals_result_ac["train"]["mlogloss"][-1]
    print(f"Logloss final activity_code : {final_ac:.5f}")
    print("=== Modèle activity_code entraîné ===\n")

    print("=== Entraînement modèle 2/3 : trial_has_fall ===")
    params_trial = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    evals_result_trial = {}
    dtrain_trial = xgb.DMatrix(X_train, label=y_trial)
    booster_trial = xgb.train(
        params=params_trial,
        dtrain=dtrain_trial,
        num_boost_round=num_rounds,
        evals=[(dtrain_trial, "train")],
        evals_result=evals_result_trial,
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, "trial_has_fall")]
    )
    final_trial = evals_result_trial["train"]["logloss"][-1]
    print(f"Logloss final trial_has_fall : {final_trial:.5f}")
    print("=== Modèle trial_has_fall entraîné ===\n")

    print("=== Entraînement modèle 3/3 : is_fall ===")
    params_isfall = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    evals_result_isfall = {}
    dtrain_isfall = xgb.DMatrix(X_train, label=y_isfall)
    booster_isfall = xgb.train(
        params=params_isfall,
        dtrain=dtrain_isfall,
        num_boost_round=num_rounds,
        evals=[(dtrain_isfall, "train")],
        evals_result=evals_result_isfall,
        verbose_eval=False,
        callbacks=[TqdmCallback(num_rounds, "is_fall")]
    )
    final_isfall = evals_result_isfall["train"]["logloss"][-1]
    print(f"Logloss final is_fall : {final_isfall:.5f}")
    print("=== Modèle is_fall entraîné ===\n")

    return booster_ac, booster_trial, booster_isfall

def per_row_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = (y_true == y_pred).mean() if y_true.size else 0.0
    return float(acc)

def per_trial_metrics_activity(test_df, y_pred_ac, id_to_ac):
    df = test_df.copy()
    df["pred_ac_id"] = y_pred_ac
    df["pred_activity_code"] = df["pred_ac_id"].map(id_to_ac)
    correct = 0
    total = 0
    gb = df.groupby(["user_id", "trial_id", "activity_code"], sort=False)
    for (_, _, _), g in tqdm(gb, total=gb.ngroups, desc="Éval activité par essai", unit="trial"):
        true_ac = str(g["activity_code"].iloc[0])
        pred_counts = Counter(g["pred_activity_code"])
        if not pred_counts:
            continue
        pred_ac = pred_counts.most_common(1)[0][0]
        if pred_ac == true_ac:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def per_trial_metrics_binary(test_df, y_pred_bin, col_name, desc):
    df = test_df.copy()
    df["pred_bin"] = y_pred_bin
    correct = 0
    total = 0
    gb = df.groupby(["user_id", "trial_id", "activity_code"], sort=False)
    for (_, _, _), g in tqdm(gb, total=gb.ngroups, desc=desc, unit="trial"):
        true_val = int(g[col_name].iloc[0])
        pred_counts = Counter(g["pred_bin"])
        if not pred_counts:
            continue
        pred_val = pred_counts.most_common(1)[0][0]
        if pred_val == true_val:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def save_models(booster_ac, booster_trial, booster_isfall):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "xgboost_activity_code.pkl", "wb") as f:
        pickle.dump(booster_ac, f)
    with open(MODELS_DIR / "xgboost_trial_has_fall.pkl", "wb") as f:
        pickle.dump(booster_trial, f)
    with open(MODELS_DIR / "xgboost_is_fall.pkl", "wb") as f:
        pickle.dump(booster_isfall, f)
    print(f"Modèles sauvegardés dans le dossier: {MODELS_DIR.resolve()}")

def main():
    train_df, test_df = load_data()
    feature_cols = build_feature_columns(train_df)
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_ac, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    y_train_trial, y_test_trial, y_train_isfall, y_test_isfall = build_targets(train_df, test_df)
    n_classes = len(ac_to_id)

    booster_ac, booster_trial, booster_isfall = train_models(
        X_train, y_train_ac, y_train_trial, y_train_isfall, n_classes
    )

    save_models(booster_ac, booster_trial, booster_isfall)

    print("=== Prédiction sur le jeu de test ===")
    dtest = xgb.DMatrix(X_test)
    probs_ac = booster_ac.predict(dtest)
    y_pred_ac = np.argmax(probs_ac, axis=1)
    probs_trial = booster_trial.predict(dtest)
    y_pred_trial = (probs_trial >= 0.5).astype(int)
    probs_isfall = booster_isfall.predict(dtest)
    y_pred_isfall = (probs_isfall >= 0.5).astype(int)
    print("=== Prédictions terminées ===\n")

    print("=== Résultats modèle activité (activity_code) ===")
    mask_valid_ac = y_test_ac >= 0
    row_acc_ac = per_row_metrics(y_test_ac[mask_valid_ac], y_pred_ac[mask_valid_ac])
    trial_acc_ac = per_trial_metrics_activity(test_df, y_pred_ac, id_to_ac)
    print(f"Accuracy par ligne (activity_code) : {row_acc_ac:.4f}")
    print(f"Accuracy par essai (activity_code) : {trial_acc_ac:.4f}\n")

    print("=== Résultats modèle essai avec chute (trial_has_fall) ===")
    row_acc_trial = per_row_metrics(y_test_trial, y_pred_trial)
    trial_acc_trial = per_trial_metrics_binary(
        test_df, y_pred_trial, "trial_has_fall", "Éval trial_has_fall par essai"
    )
    print(f"Accuracy par ligne (trial_has_fall) : {row_acc_trial:.4f}")
    print(f"Accuracy par essai (trial_has_fall) : {trial_acc_trial:.4f}\n")

    print("=== Résultats modèle timestamp en chute (is_fall) ===")
    row_acc_isfall = per_row_metrics(y_test_isfall, y_pred_isfall)
    test_df_is = test_df.copy()
    test_df_is["pred_isfall"] = y_pred_isfall
    tp = ((test_df_is["is_fall"] == 1) & (test_df_is["pred_isfall"] == 1)).sum()
    tn = ((test_df_is["is_fall"] == 0) & (test_df_is["pred_isfall"] == 0)).sum()
    fp = ((test_df_is["is_fall"] == 0) & (test_df_is["pred_isfall"] == 1)).sum()
    fn = ((test_df_is["is_fall"] == 1) & (test_df_is["pred_isfall"] == 0)).sum()
    print(f"Accuracy par ligne (is_fall) : {row_acc_isfall:.4f}")
    print(f"Matrice de confusion is_fall (tp, tn, fp, fn) : {int(tp)}, {int(tn)}, {int(fp)}, {int(fn)}\n")

if __name__ == "__main__":
    main()
