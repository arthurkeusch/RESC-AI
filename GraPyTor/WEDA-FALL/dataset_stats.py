import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATH = Path("weda_all_interp.csv")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df["activity_code"] = df["activity_code"].astype(str)
    if "is_fall" in df.columns:
        df["is_fall"] = pd.to_numeric(df["is_fall"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("Column 'is_fall' is missing in the CSV file.")
    return df


def build_activity_counts(df: pd.DataFrame) -> pd.Series:
    counts = df["activity_code"].value_counts().sort_index()
    return counts


def build_colors(labels):
    d_labels = [lab for lab in labels if lab.upper().startswith("D")]
    f_labels = [lab for lab in labels if lab.upper().startswith("F")]
    d_labels_sorted = sorted(d_labels)
    f_labels_sorted = sorted(f_labels)
    n_d = len(d_labels_sorted)
    n_f = len(f_labels_sorted)

    colors = []
    for lab in labels:
        if lab.upper().startswith("D"):
            if n_d <= 1:
                val = 0.7
            else:
                idx = d_labels_sorted.index(lab)
                val = 0.3 + 0.7 * (idx / max(1, n_d - 1))
            colors.append(plt.cm.Blues(val))
        elif lab.upper().startswith("F"):
            if n_f <= 1:
                val = 0.7
            else:
                idx = f_labels_sorted.index(lab)
                val = 0.3 + 0.7 * (idx / max(1, n_f - 1))
            colors.append(plt.cm.Reds(val))
        else:
            colors.append("gray")
    return colors


def plot_activity_pie(counts: pd.Series):
    labels = counts.index.tolist()
    sizes = counts.values
    colors = build_colors(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False
    )
    ax.set_title("Data Distribution by Activity")
    ax.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_fall_pies(df: pd.DataFrame):
    df_f = df[df["activity_code"].str.upper().str.startswith("F")].copy()

    if df_f.empty:
        print("No F* activities found in the dataset.")
        return

    total_f = len(df_f)
    fall_f = (df_f["is_fall"] == 1).sum()
    non_fall_f = total_f - fall_f

    total_all = len(df)
    fall_all = (df["is_fall"] == 1).sum()
    non_fall_all = total_all - fall_all

    print("=== is_fall statistics ===")
    print(f"Total rows (F*)           : {total_f}")
    print(f" - is_fall = 1 (F*)       : {fall_f} ({fall_f / total_f * 100:.2f} %)")
    print(f" - is_fall = 0 (F*)       : {non_fall_f} ({non_fall_f / total_f * 100:.2f} %)")
    print(f"Total rows (global D+F)   : {total_all}")
    print(f" - is_fall = 1 (global)   : {fall_all} ({fall_all / total_all * 100:.4f} %)")
    print(f" - is_fall = 0 (global)   : {non_fall_all} ({non_fall_all / total_all * 100:.4f} %)\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].pie(
        [non_fall_f, fall_f],
        labels=["No fall (is_fall=0)", "Fall (is_fall=1)"],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=["#6baed6", "#fb6a4a"]
    )
    axes[0].set_title("Fall Distribution\non F* activities only")

    axes[1].pie(
        [non_fall_all, fall_all],
        labels=["No fall (is_fall=0)", "Fall (is_fall=1)"],
        autopct="%1.3f%%",
        startangle=90,
        counterclock=False,
        colors=["#9ecae1", "#de2d26"]
    )
    axes[1].set_title("Fall Distribution\non all activities (D + F)")

    for ax in axes:
        ax.axis("equal")

    plt.tight_layout()
    plt.show()


def main():
    df = load_data(CSV_PATH)

    counts = build_activity_counts(df)
    plot_activity_pie(counts)

    plot_fall_pies(df)


if __name__ == "__main__":
    main()
