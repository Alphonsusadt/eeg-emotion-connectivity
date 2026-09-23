"""Regenerasi 4 figur Bab 4.3 (Hasil Validasi Cross-Subject / LOSO-CV).

Sumber data: `new_data/phase_5_loso/5.2-5.6/output/*.csv` (dihasilkan oleh
notebook fase 5 -- logika perhitungan TIDAK diubah, skrip ini hanya
membaca ulang CSV yang sudah ada dan menggambar ulang dengan judul/label
bahasa Indonesia serta kanvas yang dikalibrasi untuk cetak, mengikuti pola
`regen_gc_vs_pdc_heatmap.py` (kanvas lebar ~13in @150dpi, skala cetak
~0,45 pada \\textwidth thesis ini).

Usage:
    python regen_loso_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOSO_DIR = r"D:\Skripsi\new_data\phase_5_loso"
OUT_DIR = r"D:\Skripsi\latex_skripsi\contents\chapter-4\fig"

DATASET_ORDER = ["GC", "PDC_Theta", "Spectral", "Combined", "GC+PDC_ROI+Spectral"]
DATASET_LABEL = {
    "GC": "GC", "PDC_Theta": "PDC_Theta", "Spectral": "Spektral",
    "Combined": "Combined", "GC+PDC_ROI+Spectral": "GC+PDC_ROI+Spektral",
}
MODEL_ORDER = ["LDA", "SVM_Tuned", "RF_Tuned", "XGBoost_Tuned", "Voting_Soft"]

FS_TITLE = 22
FS_AXIS = 18
FS_TICK = 16
FS_LEGEND = 15
FS_ANNOT = 13


def fig1_loso_summary():
    df = pd.read_csv(os.path.join(LOSO_DIR, "5.2_loso_all_models", "output", "loso_results.csv"))
    pivot = df.pivot_table(index="Model", columns="Dataset", values="Mean_Accuracy")
    pivot = pivot.loc[MODEL_ORDER, DATASET_ORDER]

    fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
    x = np.arange(len(MODEL_ORDER))
    width = 0.16
    colors = plt.cm.tab10(np.linspace(0, 1, len(DATASET_ORDER)))
    for i, ds in enumerate(DATASET_ORDER):
        offset = (i - (len(DATASET_ORDER) - 1) / 2) * width
        ax.bar(x + offset, pivot[ds].values * 100, width, label=DATASET_LABEL[ds], color=colors[i])

    ax.axhline(100 / 3, color="red", linestyle="--", linewidth=1.5, label="Peluang Acak (33,3%)")
    ax.set_title("Akurasi LOSO: 5 Model x 5 Varian Dataset", fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax.set_xlabel("Model", fontsize=FS_AXIS, labelpad=10)
    ax.set_ylabel("Akurasi LOSO Rata-rata (%)", fontsize=FS_AXIS, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=FS_TICK - 2, rotation=12, ha="right")
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.legend(title="Varian Dataset", fontsize=FS_LEGEND, title_fontsize=FS_LEGEND,
               loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "loso_accuracy_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig2_per_subject():
    df = pd.read_csv(os.path.join(LOSO_DIR, "5.3_loso_analysis", "output", "best_model_per_subject.csv"))
    d = df[df["Dataset"] == "GC+PDC_ROI+Spectral"].sort_values("Subject")
    model_name = d["Model"].iloc[0]
    acc = d["Accuracy"].values * 100
    subjects = [f"S{int(s):02d}" for s in d["Subject"].values]
    mean_acc = acc.mean()
    worst_idx = int(np.argmin(acc))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    colors = ["#d95f02" if i == worst_idx else "#1b9e77" for i in range(len(acc))]
    ax.bar(subjects, acc, color=colors)
    ax.axhline(100 / 3, color="gray", linestyle="--", linewidth=1.3, label="Peluang Acak (33,3%)")
    ax.axhline(mean_acc, color="navy", linestyle="--", linewidth=1.5, label=f"Rata-rata: {mean_acc:.1f}%")
    ax.set_title(f"Akurasi LOSO per Subjek -- GC+PDC_ROI+Spektral ({model_name})",
                 fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax.set_xlabel("Subjek", fontsize=FS_AXIS, labelpad=10)
    ax.set_ylabel("Akurasi (%)", fontsize=FS_AXIS, labelpad=10)
    ax.tick_params(axis="x", labelsize=FS_TICK - 2, rotation=0)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=FS_LEGEND, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "loso_per_subject_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig3_icc_distribution():
    # Kategori & ambang mengikuti teori ICC yang dikutip pada Bab 2
    # (sec:validasi_cross_subject, Kleinert dkk. 2024) -- BUKAN bin
    # ad-hoc (<0.4/0.4-0.7/>0.7) yang dipakai notebook 5.5 asal, supaya
    # konsisten dengan ambang yang benar-benar dirujuk di teks.
    icc_dir = os.path.join(LOSO_DIR, "5.5_icc_analysis", "output")
    datasets = {"GC": "icc_gc.csv", "PDC_Theta": "icc_pdc_theta.csv", "Spectral": "icc_spectral.csv"}
    icc_data = {k: pd.read_csv(os.path.join(icc_dir, v))["icc"].values for k, v in datasets.items()}

    bins = [0, 0.50, 0.75, 0.90, 1.0]
    cat_labels = ["Buruk (<0,50)", "Sedang (0,50-0,75)", "Baik (0,75-0,90)", "Sangat Baik (>=0,90)"]
    cat_colors = ["#c44e52", "#dd8452", "#55a868", "#4c72b0"]

    counts = {}
    for k, vals in icc_data.items():
        hist, _ = np.histogram(vals, bins=bins)
        counts[k] = hist
    print("Kategori ICC (ambang teori Bab 2):")
    for k, c in counts.items():
        n = len(icc_data[k])
        print(f"  {k:10s} n={n}  " + "  ".join(f"{l}={v} ({v/n*100:.1f}%)" for l, v in zip(cat_labels, c)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    ax = axes[0]
    colors = {"GC": "#4c72b0", "PDC_Theta": "#dd8452", "Spectral": "#55a868"}
    for k, vals in icc_data.items():
        ax.hist(vals, bins=20, range=(0, 0.9), alpha=0.55, label=DATASET_LABEL.get(k, k), color=colors[k])
    for b in [0.50, 0.75, 0.90]:
        ax.axvline(b, color="black", linestyle=":", linewidth=1)
    ax.set_title("Distribusi ICC per Jenis Fitur", fontsize=FS_TITLE - 2, fontweight="bold", pad=12)
    ax.set_xlabel("ICC(2,1)", fontsize=FS_AXIS, labelpad=8)
    ax.set_ylabel("Jumlah Fitur", fontsize=FS_AXIS, labelpad=8)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ds_labels = [DATASET_LABEL[k] for k in datasets]
    bottoms = np.zeros(len(datasets))
    for i, (lbl, col) in enumerate(zip(cat_labels, cat_colors)):
        vals = [counts[k][i] for k in datasets]
        ax.bar(ds_labels, vals, bottom=bottoms, label=lbl, color=col)
        bottoms += np.array(vals)
    ax.set_title("Kategori Reliabilitas ICC per Jenis Fitur", fontsize=FS_TITLE - 2, fontweight="bold", pad=12)
    ax.set_ylabel("Jumlah Fitur", fontsize=FS_AXIS, labelpad=8)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 2, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "loso_icc_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig4_normalization_impact():
    df = pd.read_csv(os.path.join(LOSO_DIR, "5.6_subject_normalization", "output",
                                    "normalization_ablation_loso.csv"))
    datasets = ["GC", "GC+PDC_ROI+Spectral"]
    ds_labels = [DATASET_LABEL[d] for d in datasets]

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    x = np.arange(len(datasets))
    width = 0.32
    for i, (norm, label, color) in enumerate([
        (False, "Tanpa Normalisasi (Raw)", "#c44e52"),
        (True, "Dengan Z-score per Subjek", "#55a868"),
    ]):
        vals, errs = [], []
        for ds in datasets:
            row = df[(df["Dataset"] == ds) & (df["Normalized"] == norm)].iloc[0]
            vals.append(row["Mean_Accuracy"] * 100)
            errs.append(row["Std_Accuracy"] * 100)
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, capsize=5, label=label, color=color)

    ax.axhline(100 / 3, color="gray", linestyle=":", linewidth=1.3, label="Peluang Acak (33,3%)")
    ax.set_title("Dampak Normalisasi Z-score per Subjek terhadap Akurasi LOSO\n(Model SVM_Tuned)",
                 fontsize=FS_TITLE - 2, fontweight="bold", pad=14)
    ax.set_xlabel("Varian Dataset", fontsize=FS_AXIS, labelpad=10)
    ax.set_ylabel("Akurasi LOSO (%)", fontsize=FS_AXIS, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 95)
    ax.legend(fontsize=FS_LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "loso_normalization_impact.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_loso_summary()
    fig2_per_subject()
    fig3_icc_distribution()
    fig4_normalization_impact()
