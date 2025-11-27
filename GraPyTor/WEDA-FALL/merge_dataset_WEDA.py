import os, re, queue
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Iterable
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from tqdm import tqdm

ROOT = Path("dataset")
FREQ_DIR = ROOT / "50Hz"
OUT_DIR = Path("weda50_by_activity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLS_ORDER = [
    "user_id", "trial_id", "activity_code",
    "trial_has_fall", "is_fall",
    "Age", "Height (m)", "Weight (Kg)", "Gender",
    "timestamp",
    "accel_x_list", "accel_y_list", "accel_z_list",
    "gyro_x_list", "gyro_y_list", "gyro_z_list",
    "orientation_s_list", "orientation_i_list", "orientation_j_list", "orientation_k_list",
    "vertical_Accel_x", "vertical_Accel_y", "vertical_Accel_z"
]

def _load_fall_timestamps() -> Dict[str, List[Tuple[float, float]]]:
    csv_path = ROOT / "fall_timestamps.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    fn_col = cols.get("filename")
    start_col = cols.get("start_time") or cols.get("starttime")
    end_col = cols.get("end_time") or cols.get("endtime")
    if fn_col is None or start_col is None or end_col is None:
        return {}
    intervals: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        fname = str(row[fn_col]).strip()
        try:
            st = float(row[start_col]); et = float(row[end_col])
        except Exception:
            continue
        if np.isnan(st) or np.isnan(et):
            continue
        if et < st:
            st, et = et, st
        intervals.setdefault(fname, []).append((st, et))
    for k in intervals:
        intervals[k].sort()
    return intervals

FALL_INTERVALS = _load_fall_timestamps()
_USERS_BY_ID: Dict[int, Dict[str, object]] = {}

def _load_users_table() -> pd.DataFrame:
    candidates = [ROOT / "users.csv", Path.cwd() / "users.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        return pd.DataFrame(columns=["User_id", "Age", "Height (m)", "Weight (Kg)", "Gender"])
    df = pd.read_csv(csv_path)
    if "User_id" in df.columns:
        df["User_id"] = pd.to_numeric(df["User_id"], errors="coerce").astype("Int64")
    else:
        for c in df.columns:
            if c.lower() in ("user_id", "userid", "id"):
                df.rename(columns={c: "User_id"}, inplace=True)
                df["User_id"] = pd.to_numeric(df["User_id"], errors="coerce").astype("Int64")
                break
    return df

def _ensure_users_loaded():
    global _USERS_BY_ID
    if _USERS_BY_ID:
        return
    users_df = _load_users_table()
    _USERS_BY_ID = {}
    if not users_df.empty and "User_id" in users_df.columns:
        for _, r in users_df.iterrows():
            uid = r["User_id"]
            if pd.isna(uid):
                continue
            _USERS_BY_ID[int(uid)] = {
                "Age": r.get("Age", np.nan),
                "Height (m)": r.get("Height (m)", np.nan),
                "Weight (Kg)": r.get("Weight (Kg)", np.nan),
                "Gender": r.get("Gender", np.nan),
            }

def _read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=None, engine="python")

def _detect_time_col(df: pd.DataFrame, prefer: List[str]) -> Optional[str]:
    for c in prefer:
        if c in df.columns:
            return c
    for c in df.columns:
        if "time" in c.lower():
            return c
    return None

def _detect_axes(df: pd.DataFrame, include_prefixes: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def find_axis(ax: str) -> Optional[str]:
        cands = []
        for col in df.columns:
            if ax.lower() not in col.lower():
                continue
            if any(pref.lower() in col.lower() for pref in include_prefixes):
                cands.append(col)
        if cands:
            cands.sort(key=lambda s: (not s.lower().endswith("_list"), len(s)))
            return cands[0]
        return None
    return find_axis("x"), find_axis("y"), find_axis("z")

def _prepare_sensor_df(df: pd.DataFrame, file_path: Optional[Path] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype(float)
    df = df.sort_values("timestamp")
    if not df["timestamp"].is_unique:
        duplicated = df[df["timestamp"].duplicated(keep=False)]
        df = df[~df["timestamp"].duplicated(keep="first")]
    return df.set_index("timestamp", drop=True)

def _read_accel_like(p: Path, prefixes: List[str], time_pref: List[str], out_base: str) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = _read_csv(p)
    tcol = _detect_time_col(df, time_pref)
    if tcol is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df[tcol], errors="coerce").astype(float)
    xcol, ycol, zcol = _detect_axes(df, include_prefixes=prefixes)
    if out_base == "accel":
        if xcol is not None: out["accel_x_list"] = pd.to_numeric(df[xcol], errors="coerce")
        if ycol is not None: out["accel_y_list"] = pd.to_numeric(df[ycol], errors="coerce")
        if zcol is not None: out["accel_z_list"] = pd.to_numeric(df[zcol], errors="coerce")
    elif out_base == "gyro":
        if xcol is not None: out["gyro_x_list"] = pd.to_numeric(df[xcol], errors="coerce")
        if ycol is not None: out["gyro_y_list"] = pd.to_numeric(df[ycol], errors="coerce")
        if zcol is not None: out["gyro_z_list"] = pd.to_numeric(df[zcol], errors="coerce")
    elif out_base == "vertical_accel":
        if xcol is not None: out["vertical_Accel_x"] = pd.to_numeric(df[xcol], errors="coerce")
        if ycol is not None: out["vertical_Accel_y"] = pd.to_numeric(df[ycol], errors="coerce")
        if zcol is not None: out["vertical_Accel_z"] = pd.to_numeric(df[zcol], errors="coerce")
    return out

def read_accel_main(p: Path) -> pd.DataFrame:
    d = _read_accel_like(p, ["accel", "accelerometer"], ["accel_time_list", "time", "accel_time"], "accel")
    keep = ["timestamp", "accel_x_list", "accel_y_list", "accel_z_list"]
    return d[keep] if not d.empty else d

def read_accel_vertical(p: Path) -> pd.DataFrame:
    d = _read_accel_like(p, ["vertical_accel", "vertical", "accel"], ["accel_time_list", "time", "accel_time"], "vertical_accel")
    keep = ["timestamp", "vertical_Accel_x", "vertical_Accel_y", "vertical_Accel_z"]
    return d[keep] if not d.empty else d

def read_gyro(p: Path) -> pd.DataFrame:
    d = _read_accel_like(p, ["gyro", "gyroscope"], ["gyro_time_list", "time", "gyro_time"], "gyro")
    keep = ["timestamp", "gyro_x_list", "gyro_y_list", "gyro_z_list"]
    return d[keep] if not d.empty else d

def read_ori(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = _read_csv(p)
    tcol = _detect_time_col(df, ["orientation_time_list", "orientation_time", "time"])
    if tcol is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df[tcol], errors="coerce").astype(float)
    for c in ["orientation_s_list", "orientation_i_list", "orientation_j_list", "orientation_k_list"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out

_ROOT_RE = re.compile(r"(?:_vertical)?_accel\.csv$", flags=re.IGNORECASE)

def _root_from_filename(fname: str) -> str:
    return _ROOT_RE.sub("", fname)

def parse_ur(base: str):
    u = re.search(r"u(\d+)", base, re.I)
    r = re.search(r"r(\d+)", base, re.I)
    user_id = int(u.group(1)) if u else 0
    trial_id = int(r.group(1)) if r else 0
    return user_id, trial_id

def _inject_user_attributes(df: pd.DataFrame, user_id: int) -> None:
    _ensure_users_loaded()
    attr = _USERS_BY_ID.get(int(user_id))
    if attr is None:
        df["Age"] = np.nan
        df["Height (m)"] = np.nan
        df["Weight (Kg)"] = np.nan
        df["Gender"] = np.nan
        return
    df["Age"] = attr.get("Age", np.nan)
    df["Height (m)"] = attr.get("Height (m)", np.nan)
    df["Weight (Kg)"] = attr.get("Weight (Kg)", np.nan)
    df["Gender"] = attr.get("Gender", np.nan)

def process_trial(activity_code: str, accel_main_path: Path) -> pd.DataFrame:
    root = _root_from_filename(accel_main_path.name)
    parent = accel_main_path.parent
    accel_main_path = parent / f"{root}_accel.csv"
    accel_vert_path = parent / f"{root}_vertical_accel.csv"
    gyro_path = parent / f"{root}_gyro.csv"
    ori_path = parent / f"{root}_orientation.csv"
    base = root
    try:
        accel_main = read_accel_main(accel_main_path)
    except Exception:
        accel_main = pd.DataFrame()
    try:
        if accel_vert_path.exists():
            accel_vert = read_accel_vertical(accel_vert_path)
        else:
            maybe_vert = read_accel_vertical(accel_main_path)
            accel_vert = maybe_vert if any(
                c in maybe_vert.columns for c in
                ["vertical_Accel_x", "vertical_Accel_y", "vertical_Accel_z"]
            ) else pd.DataFrame()
    except Exception:
        accel_vert = pd.DataFrame()
    try:
        gyro = read_gyro(gyro_path) if gyro_path.exists() else pd.DataFrame()
    except Exception:
        gyro = pd.DataFrame()
    try:
        ori = read_ori(ori_path) if ori_path.exists() else pd.DataFrame()
    except Exception:
        ori = pd.DataFrame()
    parts = []
    for part, pth in ((accel_main, accel_main_path), (accel_vert, accel_vert_path), (gyro, gyro_path), (ori, ori_path)):
        if part is not None and not part.empty:
            parts.append(_prepare_sensor_df(part, pth))
    if not parts:
        return pd.DataFrame()
    df = parts[0]
    for nxt in parts[1:]:
        if not nxt.index.is_unique:
            raise RuntimeError("Internal error: non-unique timestamp index after prepare()")
        df = df.join(nxt, how="outer")
    df = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    user_id, trial_id = parse_ur(base)
    df.insert(0, "activity_code", activity_code)
    df.insert(0, "trial_id", trial_id)
    df.insert(0, "user_id", user_id)
    _inject_user_attributes(df, user_id)
    has_fall = 1 if activity_code.upper().startswith("F") else 0
    df["trial_has_fall"] = has_fall
    trial_key = f"{activity_code}/{base}"
    intervals = FALL_INTERVALS.get(trial_key, [])
    if has_fall and intervals:
        ts = df["timestamp"].to_numpy(dtype=float)
        in_fall = np.zeros(len(df), dtype=int)
        for st, et in intervals:
            in_fall |= ((ts >= st) & (ts <= et)).astype(int)
        df["is_fall"] = in_fall
    else:
        df["is_fall"] = 0
    out = df.reindex(columns=COLS_ORDER, fill_value=np.nan)
    return out

def _iter_roots(activity_code: str) -> Iterable[str]:
    d = FREQ_DIR / activity_code
    if not d.exists() or not d.is_dir():
        return []
    files = list(d.glob("*_accel.csv")) + list(d.glob("*_vertical_accel.csv"))
    roots = sorted({_root_from_filename(p.name) for p in files})
    return roots

def process_activity(activity_code: str, progress_queue) -> None:
    d = FREQ_DIR / activity_code
    if not d.exists() or not d.is_dir():
        return
    roots = list(_iter_roots(activity_code))
    if not roots:
        return
    out_path = OUT_DIR / f"weda50_{activity_code}.csv"
    if out_path.exists():
        out_path.unlink()
    header_written = False
    seen_trials = set()
    for root in roots:
        accel_main_path = d / f"{root}_accel.csv"
        uid, rid = parse_ur(root)
        key = (uid, rid, activity_code)
        try:
            if key not in seen_trials:
                df = process_trial(activity_code, accel_main_path)
                if not df.empty:
                    df.to_csv(out_path, mode="a", header=not header_written, index=False)
                    header_written = True
                    seen_trials.add(key)
        except Exception:
            pass
        finally:
            try:
                progress_queue.put(1)
            except Exception:
                pass

def _list_activities() -> List[str]:
    activities = [f"D{str(i).zfill(2)}" for i in range(1, 12)] + [f"F{str(i).zfill(2)}" for i in range(1, 9)]
    return [ac for ac in activities if (FREQ_DIR / ac).is_dir()]

def _count_total_roots(activity_codes: List[str]) -> int:
    total = 0
    for ac in activity_codes:
        total += len(list(_iter_roots(ac)))
    return total

def user_dataset():
    data = [
        [1, 22, 1.76, 56.3, "Male"],
        [2, 22, 1.78, 56.0, "Male"],
        [3, 20, 1.73, 69.5, "Male"],
        [4, 21, 1.70, 57.1, "Female"],
        [5, 23, 1.67, 59.6, "Male"],
        [6, 22, 1.67, 69.0, "Male"],
        [7, 21, 1.78, 68.1, "Male"],
        [8, 23, 1.62, 61.0, "Female"],
        [9, 22, 1.70, 52.0, "Female"],
        [10, 23, 1.83, 77.0, "Male"],
        [11, 23, 1.69, 61.8, "Male"],
        [12, 23, 1.78, 64.5, "Female"],
        [13, 22, 1.79, 66.0, "Male"],
        [14, 46, 1.84, 83.0, "Male"],
        [21, 95, 1.70, 71.0, "Male"],
        [22, 85, 1.53, 62.0, "Female"],
        [23, 82, 1.60, 60.0, "Male"],
        [24, 81, 1.52, 63.0, "Female"],
        [25, 81, 1.73, 72.0, "Male"],
        [26, 83, 1.75, 85.0, "Male"],
        [27, 89, 1.71, 71.5, "Male"],
        [28, 88, 1.57, 52.5, "Female"],
        [29, 77, 1.60, 65.9, "Female"],
        [30, 80, 1.79, 72.0, "Male"],
        [31, 88, 1.63, 53.0, "Female"]
    ]
    df = pd.DataFrame(data, columns=["User_id", "Age", "Height (m)", "Weight (Kg)", "Gender"])
    df.to_csv("users.csv", index=False)
    print("✅ Fichier CSV créé : users.csv")

def build_weda_all():
    out_csv = Path.cwd() / "weda_all.csv"
    pd.DataFrame(columns=COLS_ORDER).to_csv(out_csv, index=False)
    csv_files = sorted(OUT_DIR.glob("*.csv"))
    total_rows = sum(max(0, sum(1 for _ in open(p, "r", encoding="utf-8", errors="ignore")) - 1) for p in csv_files)
    with tqdm(total=total_rows, desc="Fusion WEDA (rows)", unit="row") as pbar:
        for p in csv_files:
            for chunk in pd.read_csv(p, sep=None, engine="python", chunksize=20000):
                chunk = chunk.reindex(columns=COLS_ORDER, fill_value=np.nan)
                chunk.to_csv(out_csv, mode="a", header=False, index=False)
                pbar.update(len(chunk))
    print("✅ Fichier CSV créé : weda_all.csv")

def main():
    user_dataset()
    activities = _list_activities()
    if not activities:
        print("No activity directories found.")
        return
    total_roots = _count_total_roots(activities)
    if total_roots == 0:
        print("No trials found.")
        return
    max_workers = min(len(activities), os.cpu_count() or 1)
    with Manager() as manager:
        progress_queue = manager.Queue()
        with tqdm(total=total_roots, desc="WEDA (files)", unit="file") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(process_activity, ac, progress_queue) for ac in activities]
                processed = 0
                while processed < total_roots:
                    try:
                        n = progress_queue.get(timeout=0.2)
                        pbar.update(n)
                        processed += n
                    except queue.Empty:
                        if all(f.done() for f in futs):
                            while True:
                                try:
                                    n = progress_queue.get_nowait()
                                    pbar.update(n)
                                    processed += n
                                except queue.Empty:
                                    break
                            break
                for f in futs:
                    try:
                        f.result()
                    except Exception as e:
                        print(f"[ERROR] activity failed: {e}")
    build_weda_all()

if __name__ == "__main__":
    main()
