import numpy as np
import pandas as pd
from pathlib import Path

INPUT_CSV = Path("weda_all_interp.csv")
TRAIN_OUT = Path("weda_train.csv")
TEST_OUT = Path("weda_test.csv")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

df = pd.read_csv(INPUT_CSV, sep=None, engine="python")

df["activity_code"] = df["activity_code"].astype(str)
df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce").astype("Int64")

df["user_id"] = df["user_id"].astype(int)
df["trial_id"] = df["trial_id"].astype(int)

trials = df[["activity_code", "user_id", "trial_id"]].drop_duplicates().reset_index(drop=True)
trials["split"] = ""

rng = np.random.default_rng(RANDOM_SEED)
activities = sorted(trials["activity_code"].unique())

for ac in activities:
    mask_ac = trials["activity_code"] == ac
    idx_ac = trials.index[mask_ac].to_numpy()
    if idx_ac.size == 0:
        continue
    shuffled = idx_ac.copy()
    rng.shuffle(shuffled)
    n_total = shuffled.size
    n_train = int(n_total * TRAIN_RATIO)
    if n_total >= 2 and n_train == n_total:
        n_train = n_total - 1
    train_idx = shuffled[:n_train]
    test_idx = shuffled[n_train:]
    trials.loc[train_idx, "split"] = "train"
    trials.loc[test_idx, "split"] = "test"

trials.loc[trials["split"] == "", "split"] = "train"

df_split = df.merge(trials, on=["activity_code", "user_id", "trial_id"], how="inner")

train_df = df_split[df_split["split"] == "train"].drop(columns=["split"])
test_df = df_split[df_split["split"] == "test"].drop(columns=["split"])

train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)
