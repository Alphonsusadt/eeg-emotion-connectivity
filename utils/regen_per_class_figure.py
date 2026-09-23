"""Regenerasi figur Bab 4.4 (Analisis Per-Kelas Emosi / Confusion Matrix).

Sumber data: `new_data/phase_5_loso/5.7_per_class_analysis/output/
loso_predictions.csv` (confusion matrix LOSO dihitung ulang langsung dari
prediksi mentah, dicocokkan terhadap `classification_report_loso.csv` --
recall diagonal harus sama persis) dan confusion matrix Grouped 5-fold
dibaca manual dari `figures/5.7_confusion_matrix_grouped5fold.png` asli
(raw predictions untuk skema ini tidak disimpan terpisah oleh notebook
5.7, hanya gambarnya -- angka selnya sudah dicocokkan terhadap recall
pada `classification_report_grouped5fold.csv` sebelum dipakai di sini).

Kanvas dikalibrasi mengikuti pola `regen_gc_vs_pdc_heatmap.py` /
`regen_loso_figures.py` (skala cetak ~0,45 pada \\textwidth thesis ini).

Usage:
    python regen_per_class_figure.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = r"D:\Skripsi\latex_skripsi\contents\chapter-4\fig"

CLASS_NAMES = ["Negatif", "Netral", "Positif"]

# Baris = aktual, kolom = prediksi. Grouped 5-fold dibaca dari figur asli
# (dicocokkan terhadap classification_report_grouped5fold.csv: recall
# 69,78% / 68,44% / 88,44% -- cocok persis dgn diagonal/rowsum di bawah).
CM_GROUPED = np.array([
    [157, 44, 24],
    [57, 154, 14],
    [15, 11, 199],
])
# LOSO dihitung langsung dari loso_predictions.csv via sklearn.confusion_matrix,
# dicocokkan terhadap classification_report_loso.csv (recall 68,00% / 71,56% / 88,00%).
CM_LOSO = np.array([
    [153, 48, 24],
    [56, 161, 8],
    [17, 10, 198],
])

FS_TITLE = 18
FS_SUBTITLE = 15
FS_AXIS = 14
FS_TICK = 13
FS_ANNOT = 14


def plot_cm(ax, cm, normalize, cmap, show_ylabel):
    if normalize:
        data = cm / cm.sum(axis=1, keepdims=True) * 100
        fmt_annot = np.array([[f"{v:.1f}%" for v in row] for row in data])
        vmax = 100
    else:
        data = cm
        fmt_annot = cm.astype(str)
        vmax = cm.max()
    sns.heatmap(data, annot=fmt_annot, fmt="", cmap=cmap, vmin=0, vmax=vmax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=False, ax=ax, annot_kws={"size": FS_ANNOT}, linewidths=0.6)
    ax.set_xlabel("Prediksi", fontsize=FS_AXIS)
    if show_ylabel:
        ax.set_ylabel("Aktual", fontsize=FS_AXIS)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=FS_TICK, rotation=0)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5), facecolor="white")

    plot_cm(axes[0, 0], CM_GROUPED, normalize=False, cmap="Blues", show_ylabel=True)
    axes[0, 0].set_title("Jumlah (raw count)", fontsize=FS_SUBTITLE, pad=10)
    plot_cm(axes[0, 1], CM_GROUPED, normalize=True, cmap="Blues", show_ylabel=False)
    axes[0, 1].set_title("Ternormalisasi per baris (recall)", fontsize=FS_SUBTITLE, pad=10)

    plot_cm(axes[1, 0], CM_LOSO, normalize=False, cmap="Oranges", show_ylabel=True)
    plot_cm(axes[1, 1], CM_LOSO, normalize=True, cmap="Oranges", show_ylabel=False)

    fig.text(0.5, 0.965, "Grouped 5-fold -- GC+PDC_ROI+Spektral + XGBoost_Tuned",
              ha="center", fontsize=FS_TITLE, fontweight="bold")
    fig.text(0.5, 0.475, "LOSO (15-fold) -- GC+PDC_ROI+Spektral + Voting_Soft",
              ha="center", fontsize=FS_TITLE, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.subplots_adjust(hspace=0.55)

    out = os.path.join(OUT_DIR, "per_class_confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)

    for name, cm in [("Grouped", CM_GROUPED), ("LOSO", CM_LOSO)]:
        recalls = cm.diagonal() / cm.sum(axis=1)
        print(f"{name} recall check:", [f"{r:.4f}" for r in recalls])


if __name__ == "__main__":
    main()
