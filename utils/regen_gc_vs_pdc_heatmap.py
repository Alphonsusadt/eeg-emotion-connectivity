"""Regenerasi heatmap akurasi klasifikasi (Bab 4.2, GC vs PDC vs Combined).

Sumber data: `phase_4_classification/4.7_gc_vs_pdc_comparison/output/
master_comparison_table.csv` (132 baris = 12 model x 11 varian dataset,
StratifiedGroupKFold 5 fold, dihasilkan oleh notebook 4.7 -- logika
perhitungan TIDAK diubah, skrip ini hanya membaca ulang CSV yang sudah ada
dan menggambar ulang).

Figur `figures/gc_vs_pdc_comparison_heatmap.png` yang lama tidak dipakai:
judulnya bahasa Inggris dan kanvasnya 20,3x7,9 in @150dpi tanpa penyesuaian
font untuk cetak (skalanya jadi ~0,29 saat \\textwidth ~5,87 in). Kanvas di
sini memakai lebar 13 in mengikuti kalibrasi `regen_stat_figures.py` (skala
cetak ~0,45 pada \\textwidth), sehingga anotasi 20pt tercetak ~9pt -- ukuran
yang sama yang sudah diverifikasi terbaca pada `significance_heatmap.png`.

Usage:
    python regen_gc_vs_pdc_heatmap.py
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = (r"D:\Skripsi\new_data\phase_4_classification\4.7_gc_vs_pdc_comparison"
            r"\output\master_comparison_table.csv")
OUT_DIR = r"D:\Skripsi\new_data\phase_4_classification\4.7_gc_vs_pdc_comparison\figures"

# Urutan kolom: dari fitur tunggal (paling lemah) ke kombinasi (paling kuat),
# supaya pola "menguat ke kanan" langsung terlihat tanpa perlu sort manual.
DATASET_ORDER = [
    "GC", "GC_ROI", "PDC_Theta", "PDC_ROI", "Spectral",
    "GC+Spectral", "PDC_Theta+Spectral", "GC+PDC_ROI",
    "GC+PDC_Theta+Spectral", "GC+PDC_ROI+Spectral", "Combined",
]

FS_TITLE = 24
FS_AXIS = 20
FS_TICK = 20
FS_ANNOT = 20
FS_CBAR = 16


def main():
    df = pd.read_csv(CSV_PATH)
    pivot = df.pivot_table(index="Model", columns="Dataset", values="Mean_Accuracy")
    pivot = pivot[DATASET_ORDER]
    # Baris diurutkan dari model terkuat (rata-rata lintas 11 dataset) ke terlemah.
    row_order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    annot = (pivot * 100).round(1).astype(str)

    fig, ax = plt.subplots(figsize=(13, 9), facecolor="white")
    sns.heatmap(pivot, annot=annot, fmt="", cmap="Blues", linewidths=0.8,
                vmin=0.45, vmax=0.80, ax=ax,
                annot_kws={"size": FS_ANNOT}, cbar_kws={"shrink": 0.85,
                "label": "Akurasi rata-rata (%)"})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_CBAR)
    cbar.set_label("Akurasi rata-rata (%)", fontsize=FS_AXIS - 2)

    ax.set_title("Akurasi Klasifikasi Rata-rata: 12 Model x 11 Varian Dataset",
                  fontsize=FS_TITLE, fontweight="bold", pad=14)
    ax.set_xlabel("Varian Dataset", fontsize=FS_AXIS, labelpad=10)
    ax.set_ylabel("Model", fontsize=FS_AXIS, labelpad=10)
    ax.tick_params(axis="y", rotation=0, labelsize=FS_TICK)
    ax.tick_params(axis="x", rotation=40, labelsize=FS_TICK)
    plt.setp(ax.get_xticklabels(), ha="right")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "gc_vs_pdc_accuracy_heatmap_id.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)
    print(pivot.round(4))


if __name__ == "__main__":
    main()
