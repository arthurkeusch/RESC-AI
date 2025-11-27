import pandas as pd
from pathlib import Path
from tqdm import tqdm

IN_PATH = Path("") / "weda_all.csv"
OUT_PATH = Path("") / "weda_all_interp.csv"

def _count_lines(p: Path, buf=1024 * 1024):
    n = 0
    with p.open("rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            n += b.count(b"\n")
    return max(0, n - 1)

total_lines = _count_lines(IN_PATH)

chunks = []
for chunk in tqdm(pd.read_csv(IN_PATH, sep=None, engine="python", chunksize=5000), total=(total_lines // 5000 + 1), desc="Chargement CSV", unit="chunk"):
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df = df.sort_values(["user_id", "trial_id", "activity_code", "timestamp"], kind="mergesort")

exclude_cols = {
    "user_id", "trial_id", "activity_code", "trial_has_fall", "is_fall",
    "Age", "Height (m)", "Weight (Kg)", "Gender",
    "timestamp", "accel_time_list", "gyro_time_list", "orientation_time_list"
}
signal_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in "fc"]

def _decimals_for_series(s: pd.Series) -> int:
    v = s.dropna().astype(float)
    if v.size < 2:
        return 1
    diffs = v.diff().abs().dropna()
    if diffs.empty:
        return 1
    step = float(diffs.median())
    return 2 if step < 0.05 else 1

def _interp_and_round_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("timestamp", kind="mergesort")
    for c in signal_cols:
        s = g[c]
        if s.notna().any():
            g[c] = s.interpolate(method="linear", limit=9, limit_area="inside")
            nd = _decimals_for_series(s)
            nd = 2 if nd > 2 else (1 if nd < 1 else nd)
            g[c] = g[c].round(nd)
    return g

with OUT_PATH.open("w", encoding="utf-8", newline=""):
    pass

gby = df.groupby(["user_id", "trial_id", "activity_code"], sort=False, group_keys=False)
total_groups = gby.ngroups
header_written = False

for (uid, tid, ac), g in tqdm(gby, total=total_groups, desc="Interpolation", unit="group"):
    g = _interp_and_round_group(g)
    g.to_csv(OUT_PATH, mode="a", header=not header_written, index=False)
    header_written = True
