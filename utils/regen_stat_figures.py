"""Regenerasi tiga figur statistik Bab 4 agar terbaca pada ukuran cetak.

Hasil audit halaman PDF (27 Juli 2026) menemukan tiga figur yang teksnya
menyusut di bawah ~5 pt saat dicetak:

  Gambar 4.1  all_methods_total_strength_boxplot.png
  Gambar 4.3  gc_threshold_comparison.png
  Gambar 4.12 gc_zscore_heatmap.png + gc_individual_lines.png

Dua perubahan pokok:

1. Kanvas diperkecil (bukan font dinaikkan). Pada kanvas 24 in yang dicetak
   pada \\textwidth ~5,87 in, faktor susutnya 0,24 sehingga font harus
   dinaikkan ekstrem dan tata letaknya jadi rusak. Kanvas ~13 in memberi
   faktor ~0,45 sehingga ukuran font wajar sudah cukup.

2. Metrik in/out dibuang dari figur individu. `mean_out_degree` dan
   `mean_in_degree` IDENTIK secara matematis (rata-rata derajat keluar lintas
   kanal selalu sama dengan derajat masuk), begitu pula strength, dan
   total = out + in. Jadi enam kolom lama hanya memuat dua informasi; setelah
   di-z-score ketiga kolom degree persis sama. Menyisakan Total Degree dan
   Total Strength membuat sel jauh lebih besar sekaligus lebih jujur.

Usage:
    python regen_stat_figures.py
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_DIR = r"D:\Skripsi\new_data\phase_1_data_structuring\csv"
GC_MAT = r"D:\Skripsi\new_data\01_granger_causality\output\gc_matrices"
OUT_CLASS = r"D:\Skripsi\new_data\phase_2_statistics\2.2_class_analysis\figures"
OUT_INDIV = (r"D:\Skripsi\new_data\phase_2_statistics"
             r"\2.1_individual_analysis\figures")
OUT_GC = r"D:\Skripsi\new_data\01_granger_causality\output"

METHODS = [("gc", "GC"), ("pdc_delta", "PDC Delta"), ("pdc_theta", "PDC Theta"),
           ("pdc_alpha", "PDC Alpha"), ("pdc_beta", "PDC Beta"),
           ("pdc_gamma", "PDC Gamma"), ("pdc_broadband", "PDC Broadband")]
EMOTIONS = ["negative", "neutral", "positive"]
PALETTE = {"negative": "#E15759", "neutral": "#4E79A7", "positive": "#59A14F"}

FS_SUP, FS_TITLE, FS_LABEL, FS_TICK = 26, 21, 18, 16


def load(prefix):
    return pd.read_csv(os.path.join(CSV_DIR, f"cleaned_graph_metrics_{prefix}.csv"))


def fig_boxplot():
    """Sebaran total-strength tiap metode. 3x3 (7 terpakai) supaya tiap panel
    lebih besar daripada tata letak 4+3 yang lama."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), facecolor="white")
    flat = axes.ravel()
    for ax, (prefix, label) in zip(flat, METHODS):
        df = load(prefix)
        sns.boxplot(data=df, x="class_label", y=f"{prefix}_mean_total_strength",
                    order=EMOTIONS, hue="class_label", hue_order=EMOTIONS,
                    palette=PALETTE, legend=False, ax=ax, fliersize=2)
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Total strength", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
    for ax in flat[len(METHODS):]:
        ax.axis("off")
    fig.suptitle("Sebaran Total-Strength per Kelas Emosi",
                 fontsize=FS_SUP, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(OUT_CLASS, "all_methods_total_strength_boxplot.png")
    os.makedirs(OUT_CLASS, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def _subject_matrix(df, metric):
    """Rata-rata metrik per subjek x emosi, dibakukan memakai rerata dan
    simpangan baku RATA-RATA PER SUBJEK (bukan seluruh sel).

    Normalisasi ini sengaja disamakan dengan angka yang dikutip di teks
    Subbab 4.1.6 (S12 Z = +2,27; S10 Z = -1,43), yang dihitung dari 15
    rata-rata per subjek. Kalau dibakukan atas seluruh 45 sel, nilainya
    mengecil dan pembaca tidak bisa merekonsiliasi gambar dengan teks.
    """
    piv = (df.groupby(["subject_num", "class_label"])[metric].mean()
             .unstack()[EMOTIONS])
    piv.index = [f"S{i:02d}" for i in piv.index]
    subj_mean = piv.mean(axis=1)
    return (piv - subj_mean.mean()) / subj_mean.std()


def fig_individual():
    df = load("gc")
    metrics = [("gc_mean_total_degree", "Total Degree"),
               ("gc_mean_total_strength", "Total Strength")]

    # --- peta panas skor baku ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), facecolor="white")
    for ax, (col, label) in zip(axes, metrics):
        z = _subject_matrix(df, col)
        sns.heatmap(z, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                    annot_kws={"fontsize": FS_TICK}, linewidths=0.5,
                    cbar_kws={"shrink": 0.7})
        ax.collections[0].colorbar.ax.tick_params(labelsize=FS_TICK)
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Subjek" if ax is axes[0] else "", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        plt.setp(ax.get_yticklabels(), rotation=0)
    fig.suptitle("Skor Baku Metrik Graf GC per Subjek",
                 fontsize=FS_SUP, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_INDIV, "gc_zscore_heatmap.png")
    os.makedirs(OUT_INDIV, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved:", out)

    # --- lintasan per subjek ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")
    cmap = plt.colormaps.get_cmap("tab20")
    for ax, (col, label) in zip(axes, metrics):
        piv = (df.groupby(["subject_num", "class_label"])[col].mean()
                 .unstack()[EMOTIONS])
        for i, (subj, row) in enumerate(piv.iterrows()):
            ax.plot(EMOTIONS, row.values, marker="o", markersize=6,
                    linewidth=1.8, color=cmap(i % 20), label=f"S{subj:02d}")
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Kelas emosi", fontsize=FS_LABEL)
        ax.set_ylabel("Nilai", fontsize=FS_LABEL)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(alpha=0.25)
    # Legenda 15 subjek diletakkan di bawah sebagai satu baris supaya tidak
    # menyusut menjadi kotak mini di sudut seperti versi lama.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
               fontsize=FS_TICK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Lintasan Metrik Graf GC per Subjek",
                 fontsize=FS_SUP, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    out = os.path.join(OUT_INDIV, "gc_individual_lines.png")
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def fig_threshold():
    """Empat strategi ambang, disusun 2x2 (bukan 1x4) supaya tiap matriks
    jauh lebih besar pada lebar cetak yang sama."""
    d = os.path.join(GC_MAT, "subject_13", "session_20140527")
    raw = np.load(os.path.join(d, "gc_raw_trial_01.npy"))
    pval = np.load(os.path.join(d, "p_values_trial_01.npy"))
    n = raw.shape[0]

    def density(m):
        return np.count_nonzero(m) / (n * (n - 1))

    work = raw.copy()
    np.fill_diagonal(work, 0)
    uncorr = np.where(pval < 0.05, work, 0)
    bonf = np.where(pval < 0.05 / (n * (n - 1)), work, 0)
    thr = np.percentile(work[work > 0], 90)
    prop = np.where(work >= thr, work, 0)
    panels = [("Tanpa ambang", work), ("p < 0,05 tanpa koreksi", uncorr),
              ("Koreksi Bonferroni", bonf), ("Persentil ke-90", prop)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), facecolor="white")
    for ax, (title, m) in zip(axes.ravel(), panels):
        sns.heatmap(m, cmap="hot", ax=ax, xticklabels=False,
                    yticklabels=False, cbar_kws={"shrink": 0.8})
        ax.collections[0].colorbar.ax.tick_params(labelsize=FS_TICK)
        ax.set_title(f"{title}\nDensitas = {density(m):.3f}",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Kanal sumber", fontsize=FS_LABEL)
        ax.set_ylabel("Kanal target", fontsize=FS_LABEL)
    fig.suptitle("Perbandingan Strategi Pengambangan Konektivitas GC",
                 fontsize=FS_SUP, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_GC, "gc_threshold_comparison.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    fig_boxplot()
    fig_individual()
    fig_threshold()
