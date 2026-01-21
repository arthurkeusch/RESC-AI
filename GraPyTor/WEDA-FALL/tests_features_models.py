import re
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt

TRAIN_PATH = Path("weda_train.csv")
MODELS_DIR = Path("models")
OUT_DIR = Path("feature_importances_out")

IMPORTANCE_TYPES = ["gain", "weight", "cover"]
TOP_N = 30
GLOBAL_TOP_N = 40
DPI = 160

class VotingEnsemble:
    def __init__(self, task, boosters, mode="soft", threshold=0.5, weights=None, meta=None):
        self.task = task
        self.boosters = boosters
        self.mode = mode
        self.threshold = float(threshold)
        self.weights = weights if weights is not None else {k: 1.0 for k in boosters.keys()}
        self.meta = meta if meta is not None else {}

class StackingEnsemble:
    def __init__(self, task, boosters, meta=None):
        self.task = task
        self.boosters = boosters
        self.meta = meta if meta is not None else {}
        self.w = None
        self.W = None
        self.b = 0.0
        self.threshold = 0.5
        self.n_classes = int(self.meta.get("n_classes", 0))

def load_data():
    df = pd.read_csv(TRAIN_PATH, sep=None, engine="python")
    df["activity_code"] = df["activity_code"].astype(str)
    return df

def build_feature_columns(df):
    exclude_targets = {"activity_code", "trial_has_fall", "is_fall"}
    return [c for c in df.columns if c not in exclude_targets and df[c].dtype.kind in "fc"]

def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_booster(obj):
    if isinstance(obj, xgb.Booster):
        return obj
    if hasattr(obj, "get_booster"):
        try:
            return obj.get_booster()
        except Exception:
            return None
    return None

def discover_all_models():
    return sorted([p for p in MODELS_DIR.glob("xgboost*.pkl")], key=lambda p: p.name.lower())

def score_to_vector(score_dict, feature_names):
    out = np.zeros(len(feature_names), dtype=float)
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for k, v in score_dict.items():
        if k in name_to_idx:
            out[name_to_idx[k]] = float(v)
            continue
        m = re.fullmatch(r"f(\d+)", str(k))
        if m:
            i = int(m.group(1))
            if 0 <= i < len(out):
                out[i] = float(v)
    return out

def booster_importance(booster: xgb.Booster, feature_names, importance_type: str):
    try:
        booster.feature_names = feature_names
    except Exception:
        pass
    score = booster.get_score(importance_type=importance_type)
    return score_to_vector(score, feature_names)

def extract_boosters_and_weights(obj):
    b = to_booster(obj)
    if b is not None:
        return [("booster", b, 1.0)]

    if hasattr(obj, "boosters") and isinstance(getattr(obj, "boosters"), dict):
        boosters = getattr(obj, "boosters")
        weights = getattr(obj, "weights", None)
        items = []
        for k, v in boosters.items():
            bb = to_booster(v)
            if bb is None:
                continue
            w = 1.0
            if isinstance(weights, dict):
                try:
                    w = float(weights.get(k, 1.0))
                except Exception:
                    w = 1.0
            items.append((str(k), bb, w))
        return items

    return []

def aggregate_importances(boosters_with_w, feature_names, importance_type):
    if not boosters_with_w:
        return np.zeros(len(feature_names), dtype=float)

    acc = np.zeros(len(feature_names), dtype=float)
    wsum = 0.0
    for _, booster, w in boosters_with_w:
        vec = booster_importance(booster, feature_names, importance_type)
        acc += vec * float(w)
        wsum += float(w)
    if wsum > 0:
        acc /= wsum
    return acc

def topk(vec, feature_names, k):
    idx = np.argsort(vec)[::-1]
    idx = idx[vec[idx] > 0]
    idx = idx[:k]
    return idx, vec[idx], [feature_names[i] for i in idx]

def plot_one_figure_three_panels(model_name, feature_names, vecs_by_type):
    fig, axes = plt.subplots(1, 3, figsize=(19, 7), constrained_layout=True)
    fig.suptitle(model_name, fontsize=14)

    for ax, t in zip(axes, IMPORTANCE_TYPES):
        vec = np.asarray(vecs_by_type[t], dtype=float)
        s = float(np.sum(vec))
        if s > 0:
            vec = vec / s

        idx, vals, labs = topk(vec, feature_names, TOP_N)

        if len(idx) == 0:
            ax.set_title(t)
            ax.text(0.5, 0.5, "Aucune importance non-nulle", ha="center", va="center")
            ax.set_axis_off()
            continue

        y = np.arange(len(vals))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{t} (top {min(TOP_N, len(vals))})")
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    return fig

def safe_filename(s):
    s = re.sub(r"[^\w\-. ]+", "_", s)
    s = s.strip().replace(" ", "_")
    return s[:220] if len(s) > 220 else s

def normalize(vec):
    s = float(np.sum(vec))
    if s <= 0:
        return vec.astype(float)
    return (vec / s).astype(float)

def build_global_summary(per_model_vecs_norm, feature_names):
    n_models = len(per_model_vecs_norm)
    if n_models == 0:
        return None

    avg = {t: np.zeros(len(feature_names), dtype=float) for t in IMPORTANCE_TYPES}
    nonzero_counts = {t: np.zeros(len(feature_names), dtype=int) for t in IMPORTANCE_TYPES}

    for _, vecs in per_model_vecs_norm.items():
        for t in IMPORTANCE_TYPES:
            v = vecs[t]
            avg[t] += v
            nonzero_counts[t] += (v > 0).astype(int)

    for t in IMPORTANCE_TYPES:
        avg[t] /= max(n_models, 1)

    combined = np.zeros(len(feature_names), dtype=float)
    for t in IMPORTANCE_TYPES:
        combined += avg[t]
    combined /= float(len(IMPORTANCE_TYPES))

    df = pd.DataFrame({
        "feature": feature_names,
        "avg_gain": avg["gain"],
        "avg_weight": avg["weight"],
        "avg_cover": avg["cover"],
        "avg_combined": combined,
        "models_nonzero_gain": nonzero_counts["gain"],
        "models_nonzero_weight": nonzero_counts["weight"],
        "models_nonzero_cover": nonzero_counts["cover"],
    })

    df = df.sort_values("avg_combined", ascending=False).reset_index(drop=True)
    return df

def plot_global_summary(df_summary, out_dir: Path):
    top = df_summary.head(GLOBAL_TOP_N).copy()
    fig, ax = plt.subplots(figsize=(14, 8), dpi=DPI)

    y = np.arange(len(top))
    ax.barh(y, top["avg_combined"].to_numpy(dtype=float))
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f"Moyenne des importances (gain/weight/cover) — Top {len(top)}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()

    out_png = out_dir / "GLOBAL__avg_feature_importance.png"
    fig.savefig(out_png, dpi=DPI)
    plt.show()
    plt.close(fig)

def write_text_synthesis(df_summary: pd.DataFrame, out_dir: Path, n_models: int):
    top = df_summary.head(25).copy()
    lines = []
    lines.append(f"SYNTHESE FEATURE IMPORTANCE (moyenne sur {n_models} modèles)")
    lines.append("")
    lines.append("Méthode:")
    lines.append("- Pour chaque modèle, on calcule les importances XGBoost (gain/weight/cover).")
    lines.append("- Chaque vecteur est normalisé (somme = 1) par type pour rendre les modèles comparables.")
    lines.append("- On moyenne ensuite les importances sur tous les modèles.")
    lines.append("- Score combiné = moyenne(avg_gain, avg_weight, avg_cover).")
    lines.append("")
    lines.append("Top features globales (score combiné):")
    for i, r in top.iterrows():
        lines.append(
            f"{i+1:02d}. {r['feature']} | combined={r['avg_combined']:.6f} | "
            f"gain={r['avg_gain']:.6f} | weight={r['avg_weight']:.6f} | cover={r['avg_cover']:.6f} | "
            f"nz(g/w/c)={int(r['models_nonzero_gain'])}/{int(r['models_nonzero_weight'])}/{int(r['models_nonzero_cover'])}"
        )

    lines.append("")
    lines.append("Lecture rapide:")
    lines.append("- combined élevé + nz élevé => feature utilisée souvent ET de façon consistante.")
    lines.append("- combined élevé + nz faible => feature très importante dans quelques modèles seulement (effet de corrélation/variance possible).")
    lines.append("- Si une feature est forte en gain mais faible en weight, elle est utilisée peu souvent mais avec des splits très utiles.")
    lines.append("- Si une feature est forte en weight mais faible en gain, elle est beaucoup utilisée mais apporte peu par split.")
    lines.append("")

    out_txt = out_dir / "SYNTHESE_feature_importance.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_data()
    feature_names = build_feature_columns(train_df)

    model_paths = discover_all_models()

    per_model_vecs_norm = {}
    per_model_figs_done = 0

    for path in model_paths:
        obj = load_pickle(path)
        boosters_with_w = extract_boosters_and_weights(obj)
        if not boosters_with_w:
            continue

        vecs_raw = {}
        for t in IMPORTANCE_TYPES:
            vecs_raw[t] = aggregate_importances(boosters_with_w, feature_names, t)

        vecs_norm = {t: normalize(vecs_raw[t]) for t in IMPORTANCE_TYPES}
        per_model_vecs_norm[path.name] = vecs_norm

        fig = plot_one_figure_three_panels(path.name, feature_names, vecs_raw)
        out_png = OUT_DIR / f"{safe_filename(path.stem)}__importances.png"
        fig.savefig(out_png, dpi=DPI)
        plt.show()
        plt.close(fig)
        per_model_figs_done += 1

    df_summary = build_global_summary(per_model_vecs_norm, feature_names)
    if df_summary is None:
        return

    df_summary.to_csv(OUT_DIR / "GLOBAL__avg_feature_importance.csv", index=False, encoding="utf-8")
    plot_global_summary(df_summary, OUT_DIR)
    write_text_synthesis(df_summary, OUT_DIR, n_models=len(per_model_vecs_norm))

if __name__ == "__main__":
    main()
