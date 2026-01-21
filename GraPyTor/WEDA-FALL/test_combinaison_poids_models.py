import os
import csv
import threading
import queue as thq
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import xgboost as xgb
from tqdm import tqdm
import multiprocessing as mp

TRAIN_PATH = Path("weda_train.csv")
TEST_PATH = Path("weda_test.csv")

ROUNDS = 100
THRESHOLD = 0.5

TASKS_ENABLED = {
    "activity_code": False,
    "trial_has_fall": True,
    "is_fall": False,
}

GPU_SHARE = 0.5
PREFETCH = 4

OUT_CSV = Path("xgboost_feature_poids.csv")

WEIGHT_COL = "Weight (Kg)"


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


def load_data():
    train_df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    test_df = pd.read_csv(TEST_PATH, sep=None, engine="python")
    train_df["activity_code"] = train_df["activity_code"].astype(str)
    test_df["activity_code"] = test_df["activity_code"].astype(str)
    return train_df, test_df


def build_feature_columns(df):
    exclude = {"user_id", "trial_id", "activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude and df[c].dtype.kind in "fc"]


def encode_activity(train_df, test_df):
    ac_train = sorted(train_df["activity_code"].unique())
    ac_to_id = {ac: i for i, ac in enumerate(ac_train)}
    id_to_ac = {i: ac for ac, i in ac_to_id.items()}
    y_train = train_df["activity_code"].map(ac_to_id).astype(int).to_numpy()
    y_test = test_df["activity_code"].map(ac_to_id).fillna(-1).astype(int).to_numpy()
    return y_train, y_test, ac_to_id, id_to_ac


def eval_activity_trial_level(booster, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, feature_names):
    dm = xgb.DMatrix(X_test, feature_names=feature_names)
    probs = booster.predict(dm)
    y_pred_line = np.argmax(probs, axis=1).astype(int)
    mask = y_test_ac >= 0
    row_acc = float((y_test_ac[mask] == y_pred_line[mask]).mean()) if int(mask.sum()) else 0.0
    df = test_df.copy()
    df["pred_ac"] = y_pred_line
    df["pred_code"] = df["pred_ac"].map(id_to_ac)
    y_true_trials = []
    y_pred_trials = []
    for (_, _, ac), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        true_code = str(ac)
        if true_code not in ac_to_id:
            continue
        pred_code = Counter(g["pred_code"]).most_common(1)[0][0]
        y_true_trials.append(ac_to_id[true_code])
        y_pred_trials.append(ac_to_id[str(pred_code)])
    if not y_true_trials:
        return row_acc, 0.0
    y_true_trials = np.asarray(y_true_trials, dtype=int)
    y_pred_trials = np.asarray(y_pred_trials, dtype=int)
    trial_acc = float((y_true_trials == y_pred_trials).mean())
    return row_acc, trial_acc


def eval_binary_trial_majority(booster, X_test, test_df, y_test_line, feature_names):
    dm = xgb.DMatrix(X_test, feature_names=feature_names)
    p = booster.predict(dm)
    pred_line = (p >= THRESHOLD).astype(int)
    df = test_df.copy()
    df["pred_line"] = pred_line
    df["true_line"] = y_test_line
    y_true = []
    y_pred = []
    for (_, _, _), g in df.groupby(["user_id", "trial_id", "activity_code"]):
        y_true.append(int(g["true_line"].iloc[0]))
        y_pred.append(int(Counter(g["pred_line"]).most_common(1)[0][0]))
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def eval_binary_line(booster, X_test, y_test, feature_names):
    dm = xgb.DMatrix(X_test, feature_names=feature_names)
    p = booster.predict(dm)
    y_pred = (p >= THRESHOLD).astype(int)
    return float((y_pred == y_test).mean()) if len(y_test) else 0.0


def train_activity_default(X_train, y_train_ac, n_classes, feature_names, params_extra):
    dtrain = xgb.DMatrix(X_train, label=y_train_ac, feature_names=feature_names)
    params = {"objective": "multi:softprob", "num_class": int(n_classes), "eval_metric": "mlogloss", **params_extra}
    return xgb.train(params, dtrain, num_boost_round=ROUNDS)


def train_binary_default(X_train, y_train, feature_names, params_extra):
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", **params_extra}
    return xgb.train(params, dtrain, num_boost_round=ROUNDS)


def mask_to_digits(mask_int, n):
    return format(mask_int, f"0{n}b")


def enabled_tasks_list():
    return [k for k, v in TASKS_ENABLED.items() if v]


def cpu_worker(mask_q, out_q):
    train_df, test_df = load_data()
    features = build_feature_columns(train_df)
    n = len(features)

    weight_idx = None
    if WEIGHT_COL in features:
        weight_idx = int(features.index(WEIGHT_COL))

    y_train_ac, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    y_train_trial = train_df["trial_has_fall"].astype(int).to_numpy()
    y_test_trial = test_df["trial_has_fall"].astype(int).to_numpy()
    y_train_isfall = train_df["is_fall"].astype(int).to_numpy()
    y_test_isfall = test_df["is_fall"].astype(int).to_numpy()
    n_classes = len(ac_to_id)

    X_train_full = train_df[features].to_numpy(dtype=np.float32, copy=True)
    X_test_full = test_df[features].to_numpy(dtype=np.float32, copy=True)

    X_train_full_round = None
    X_test_full_round = None
    if weight_idx is not None:
        X_train_full_round = X_train_full.copy()
        X_test_full_round = X_test_full.copy()
        X_train_full_round[:, weight_idx] = round_weight_to_half_tens(X_train_full_round[:, weight_idx])
        X_test_full_round[:, weight_idx] = round_weight_to_half_tens(X_test_full_round[:, weight_idx])

    tasks = enabled_tasks_list()
    params_cpu = {"tree_method": "hist", "nthread": 0}

    while True:
        m = mask_q.get()
        if m is None:
            out_q.put(None)
            return

        cols = [i for i in range(n) if (m >> i) & 1]
        if not cols:
            continue

        used_features = [features[i] for i in cols]
        use_rounded_weight = (weight_idx is not None) and (weight_idx in cols)

        X_train_base = X_train_full_round if use_rounded_weight else X_train_full
        X_test_base = X_test_full_round if use_rounded_weight else X_test_full

        X_train = X_train_base[:, cols]
        X_test = X_test_base[:, cols]

        row = {
            "mask_digits": mask_to_digits(m, n),
            "mask_int": int(m),
            "nb_features": int(len(cols)),
            "features": "|".join(used_features),
            "backend": "CPU",
            "tasks_done": int(len(tasks)),
            "weight_rounded": int(use_rounded_weight),
        }

        if TASKS_ENABLED["activity_code"]:
            booster_ac = train_activity_default(X_train, y_train_ac, n_classes, used_features, params_cpu)
            r, t = eval_activity_trial_level(booster_ac, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, used_features)
            row["acc_activity_row"] = float(r)
            row["acc_activity_trial"] = float(t)

        if TASKS_ENABLED["trial_has_fall"]:
            booster_thf = train_binary_default(X_train, y_train_trial, used_features, params_cpu)
            acc = eval_binary_trial_majority(booster_thf, X_test, test_df, y_test_trial, used_features)
            row["acc_trial_has_fall_trial"] = float(acc)

        if TASKS_ENABLED["is_fall"]:
            booster_is = train_binary_default(X_train, y_train_isfall, used_features, params_cpu)
            acc = eval_binary_line(booster_is, X_test, y_test_isfall, used_features)
            row["acc_is_fall_line"] = float(acc)

        out_q.put(row)


def gpu_worker(mask_q, out_q):
    train_df, test_df = load_data()
    features = build_feature_columns(train_df)
    n = len(features)

    weight_idx = None
    if WEIGHT_COL in features:
        weight_idx = int(features.index(WEIGHT_COL))

    y_train_ac, y_test_ac, ac_to_id, id_to_ac = encode_activity(train_df, test_df)
    y_train_trial = train_df["trial_has_fall"].astype(int).to_numpy()
    y_test_trial = test_df["trial_has_fall"].astype(int).to_numpy()
    y_train_isfall = train_df["is_fall"].astype(int).to_numpy()
    y_test_isfall = test_df["is_fall"].astype(int).to_numpy()
    n_classes = len(ac_to_id)

    X_train_full = train_df[features].to_numpy(dtype=np.float32, copy=True)
    X_test_full = test_df[features].to_numpy(dtype=np.float32, copy=True)

    X_train_full_round = None
    X_test_full_round = None
    if weight_idx is not None:
        X_train_full_round = X_train_full.copy()
        X_test_full_round = X_test_full.copy()
        X_train_full_round[:, weight_idx] = round_weight_to_half_tens(X_train_full_round[:, weight_idx])
        X_test_full_round[:, weight_idx] = round_weight_to_half_tens(X_test_full_round[:, weight_idx])

    tasks = enabled_tasks_list()
    params_gpu = {"tree_method": "hist", "device": "cuda"}

    local_q = thq.Queue(maxsize=max(1, int(PREFETCH)))

    def producer():
        while True:
            m = mask_q.get()
            if m is None:
                local_q.put(None)
                return
            cols = [i for i in range(n) if (m >> i) & 1]
            if not cols:
                continue
            used_features = [features[i] for i in cols]
            use_rounded_weight = (weight_idx is not None) and (weight_idx in cols)

            X_train_base = X_train_full_round if use_rounded_weight else X_train_full
            X_test_base = X_test_full_round if use_rounded_weight else X_test_full

            X_train = X_train_base[:, cols]
            X_test = X_test_base[:, cols]
            local_q.put((m, cols, used_features, X_train, X_test, use_rounded_weight))

    th = threading.Thread(target=producer, daemon=True)
    th.start()

    while True:
        item = local_q.get()
        if item is None:
            out_q.put(None)
            return

        m, cols, used_features, X_train, X_test, use_rounded_weight = item

        row = {
            "mask_digits": mask_to_digits(m, n),
            "mask_int": int(m),
            "nb_features": int(len(cols)),
            "features": "|".join(used_features),
            "backend": "GPU",
            "tasks_done": int(len(tasks)),
            "weight_rounded": int(use_rounded_weight),
        }

        try:
            if TASKS_ENABLED["activity_code"]:
                booster_ac = train_activity_default(X_train, y_train_ac, n_classes, used_features, params_gpu)
                r, t = eval_activity_trial_level(booster_ac, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, used_features)
                row["acc_activity_row"] = float(r)
                row["acc_activity_trial"] = float(t)

            if TASKS_ENABLED["trial_has_fall"]:
                booster_thf = train_binary_default(X_train, y_train_trial, used_features, params_gpu)
                acc = eval_binary_trial_majority(booster_thf, X_test, test_df, y_test_trial, used_features)
                row["acc_trial_has_fall_trial"] = float(acc)

            if TASKS_ENABLED["is_fall"]:
                booster_is = train_binary_default(X_train, y_train_isfall, used_features, params_gpu)
                acc = eval_binary_line(booster_is, X_test, y_test_isfall, used_features)
                row["acc_is_fall_line"] = float(acc)

            out_q.put(row)

        except Exception as e:
            params_cpu = {"tree_method": "hist", "nthread": 0}
            row["backend"] = "GPU_FAIL_CPU"
            row["error"] = str(e)

            if TASKS_ENABLED["activity_code"]:
                booster_ac = train_activity_default(X_train, y_train_ac, n_classes, used_features, params_cpu)
                r, t = eval_activity_trial_level(booster_ac, X_test, test_df, y_test_ac, ac_to_id, id_to_ac, used_features)
                row["acc_activity_row"] = float(r)
                row["acc_activity_trial"] = float(t)

            if TASKS_ENABLED["trial_has_fall"]:
                booster_thf = train_binary_default(X_train, y_train_trial, used_features, params_cpu)
                acc = eval_binary_trial_majority(booster_thf, X_test, test_df, y_test_trial, used_features)
                row["acc_trial_has_fall_trial"] = float(acc)

            if TASKS_ENABLED["is_fall"]:
                booster_is = train_binary_default(X_train, y_train_isfall, used_features, params_cpu)
                acc = eval_binary_line(booster_is, X_test, y_test_isfall, used_features)
                row["acc_is_fall_line"] = float(acc)

            out_q.put(row)


def load_existing_progress(csv_path: Path, expected_n: int):
    done_masks = set()
    done_trainings = 0
    if not csv_path.exists():
        return done_masks, done_trainings, False

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return set(), 0, False
            if "mask_int" not in reader.fieldnames or "mask_digits" not in reader.fieldnames:
                return set(), 0, False
            for row in reader:
                try:
                    mi = int(row.get("mask_int", ""))
                except Exception:
                    continue
                md = str(row.get("mask_digits", "") or "")
                if len(md) != expected_n:
                    return set(), 0, False
                done_masks.add(mi)
                try:
                    td = int(row.get("tasks_done", "0") or 0)
                except Exception:
                    td = 0
                done_trainings += td
        return done_masks, done_trainings, True
    except Exception:
        return set(), 0, False


def main():
    mp.set_start_method("spawn", force=True)

    train_df, _ = load_data()
    features = build_feature_columns(train_df)
    n = len(features)
    total_masks = (1 << n) - 1

    tasks = enabled_tasks_list()
    if not tasks:
        raise ValueError("Aucun modèle activé dans TASKS_ENABLED.")

    enabled_count = len(tasks)
    total_trainings = total_masks * enabled_count

    fields = [
        "mask_digits", "mask_int", "nb_features", "features", "backend", "tasks_done",
        "weight_rounded",
        "acc_activity_row", "acc_activity_trial",
        "acc_trial_has_fall_trial",
        "acc_is_fall_line",
        "error",
    ]
    fields = [f for f in fields if not (f.startswith("acc_activity") and not TASKS_ENABLED["activity_code"])
              and not (f == "acc_trial_has_fall_trial" and not TASKS_ENABLED["trial_has_fall"])
              and not (f == "acc_is_fall_line" and not TASKS_ENABLED["is_fall"])]

    done_set, done_trainings, can_resume = load_existing_progress(OUT_CSV, n)
    done_set.discard(0)

    gpu_ok = gpu_available()
    cpu_q = mp.Queue(maxsize=512)
    gpu_q = mp.Queue(maxsize=512)
    out_q = mp.Queue(maxsize=512)

    p_cpu = mp.Process(target=cpu_worker, args=(cpu_q, out_q), daemon=True)
    p_cpu.start()

    p_gpu = None
    if gpu_ok:
        p_gpu = mp.Process(target=gpu_worker, args=(gpu_q, out_q), daemon=True)
        p_gpu.start()

    gpu_every = None
    if gpu_ok:
        gpu_every = int(max(1, round(1.0 / max(1e-12, GPU_SHARE))))

    finished_workers = 0
    target_workers = 2 if gpu_ok else 1

    done_masks = len(done_set) if can_resume else 0
    done_trainings = min(done_trainings, total_trainings) if can_resume else 0

    mode = "a" if can_resume else "w"
    need_header = (mode == "w") or (not OUT_CSV.exists()) or (OUT_CSV.stat().st_size == 0)

    with open(OUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if need_header:
            writer.writeheader()

        pbar = tqdm(total=total_trainings, initial=done_trainings, dynamic_ncols=True)
        pbar.set_postfix({"masks": f"{done_masks}/{total_masks}", "tasks": enabled_count, "gpu": int(gpu_ok)})

        for mask_int in range(1, 1 << n):
            if can_resume and (mask_int in done_set):
                continue

            if gpu_ok and (mask_int % gpu_every == 0):
                gpu_q.put(mask_int)
            else:
                cpu_q.put(mask_int)

            while True:
                try:
                    row = out_q.get_nowait()
                except Exception:
                    break

                if row is None:
                    finished_workers += 1
                    continue

                writer.writerow(row)
                done_masks += 1
                td = int(row.get("tasks_done", enabled_count))
                done_trainings += td
                pbar.update(td)
                pbar.set_postfix({"masks": f"{done_masks}/{total_masks}", "tasks": enabled_count, "gpu": int(gpu_ok), "cpu": 1})

        cpu_q.put(None)
        if gpu_ok:
            gpu_q.put(None)

        while done_trainings < total_trainings:
            row = out_q.get()
            if row is None:
                finished_workers += 1
                if finished_workers >= target_workers:
                    break
                continue
            writer.writerow(row)
            done_masks += 1
            td = int(row.get("tasks_done", enabled_count))
            done_trainings += td
            pbar.update(td)
            pbar.set_postfix({"masks": f"{done_masks}/{total_masks}", "tasks": enabled_count, "gpu": int(gpu_ok), "cpu": 1})

        pbar.close()

    p_cpu.join()
    if p_gpu is not None:
        p_gpu.join()

    print(str(OUT_CSV.resolve()))


if __name__ == "__main__":
    main()
