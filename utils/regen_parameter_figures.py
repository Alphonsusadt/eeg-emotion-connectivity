"""Regenerasi empat figur pembenaran parameter Bab 4 dengan font layak cetak.

Keempatnya sebelumnya dibuat pada Juni/Juli 2026 dengan font default matplotlib
di kanvas lebar, sehingga seluruh teksnya menyusut 3-6x ketika LaTeX
memasukkannya pada \\textwidth ~ 5,87 in dan tidak terbaca saat dicetak.
Selain itu `pdc_bands_test.png` yang lama dirender 20 Juni 2026 -- SEBELUM
perbaikan matriks PDC 20 Juli 2026 -- sehingga menampilkan pola dari matriks
yang sudah tidak dipakai lagi. Versi di sini dihitung ulang dari
`02_pdc/output/pdc_matrices/` yang terkoreksi.

Logika perhitungan tidak diubah sama sekali; hanya sumber data (untuk PDC) dan
ukuran font yang disesuaikan. Sel notebook asal ikut ditambal oleh
`patch_source_notebooks.py` agar pipeline tetap reproducible.

Usage:
    python regen_parameter_figures.py
"""
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GC_DIR = r"D:\Skripsi\new_data\01_granger_causality\output\gc_matrices"
GC_FIG = r"D:\Skripsi\new_data\01_granger_causality\output\figures"
PDC_DIR = r"D:\Skripsi\new_data\02_pdc\output\pdc_matrices"
PDC_FIG = r"D:\Skripsi\new_data\02_pdc\output\figures"

# Subjek rujukan disamakan dengan topomap GC 4.1.4 dan PDC 4.1.5. Subjek 01
# (pilihan lama) tidak dipakai lagi: kanal PO6-nya mendominasi outflow PDC
# ~320x median di seluruh 15 trial sesi pertama, sehingga heatmap-nya tampak
# jenuh dan tidak mewakili data. Subjek 13 berada pada 9,2x -- wajar.
REF_SUBJECT, REF_SESSION = "subject_13", "session_20140527"
PDC_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
EMOTION_MAP = {1: "positive", 0: "neutral", -1: "negative"}
PROPORTIONAL_RETAIN = 0.10

# Kanvas lebar dicetak pada ~5,87 in, jadi seluruh font dinaikkan ~3x dari
# ukuran default matplotlib supaya tetap terbaca di kertas.
FS_SUPTITLE = 34
FS_TITLE = 27
FS_LABEL = 23
FS_TICK = 20


def compute_density(matrix):
    n = matrix.shape[0]
    return np.count_nonzero(matrix) / (n * (n - 1))


def _cbar(ax_img):
    cb = ax_img.collections[0].colorbar
    if cb is not None:
        cb.ax.tick_params(labelsize=FS_TICK)


def fig_pdc_bands():
    """PDC per pita, dihitung dari matriks TERKOREKSI (bukan render Juni)."""
    d = os.path.join(PDC_DIR, REF_SUBJECT, REF_SESSION)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white")
    for ax, band in zip(axes.flatten(), PDC_BANDS):
        m = np.load(os.path.join(d, f"pdc_{band}_trial_01.npy"))
        sns.heatmap(m, cmap="hot", ax=ax, xticklabels=False, yticklabels=False,
                    vmin=0, vmax=m.max())
        ax.set_title(f"PDC {band}", fontsize=FS_TITLE, fontweight="bold")
        _cbar(ax)
    plt.suptitle(f"PDC per Pita Frekuensi — Subjek {REF_SUBJECT[-2:]}, "
                 f"Sesi {REF_SESSION[-8:]}, Trial 1",
                 fontsize=FS_SUPTITLE, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PDC_FIG, "pdc_bands_test.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig_gc_sample():
    """Raw GC vs hasil ambang proporsional vs orde lag optimal per pasangan."""
    d = os.path.join(GC_DIR, REF_SUBJECT, REF_SESSION)
    raw = np.load(os.path.join(d, "gc_raw_trial_01.npy"))
    thr = np.load(os.path.join(d, "gc_thresholded_trial_01.npy"))
    lag = np.load(os.path.join(d, "gc_lag_trial_01.npy")).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor="white")
    panels = [("Raw GC", raw, True),
              (f"Ambang proporsional {PROPORTIONAL_RETAIN*100:.0f}%", thr, True),
              ("Orde lag optimal per pasangan", lag, False)]
    for ax, (title, mat, show_density) in zip(axes, panels):
        sns.heatmap(mat, cmap="hot", ax=ax, xticklabels=False, yticklabels=False)
        label = (f"{title}\nDensitas={compute_density(mat):.3f}"
                 if show_density else title)
        ax.set_title(label, fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Kanal sumber", fontsize=FS_LABEL)
        ax.set_ylabel("Kanal target", fontsize=FS_LABEL)
        _cbar(ax)
    plt.suptitle(f"Matriks GC Konfigurasi Optimasi — "
                 f"Subjek {REF_SUBJECT[-2:]}, Trial 1",
                 fontsize=FS_SUPTITLE, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(GC_FIG, "gc_optimized_sample.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig_gc_density():
    """Densitas graf hasil ambang: per emosi dan per subjek (675 trial)."""
    rows = []
    for subj in sorted(os.listdir(GC_DIR)):
        sp = os.path.join(GC_DIR, subj)
        if not os.path.isdir(sp):
            continue
        for sess in sorted(os.listdir(sp)):
            ssp = os.path.join(sp, sess)
            if not os.path.isdir(ssp):
                continue
            for f in sorted(os.listdir(ssp)):
                if "thresholded" in f and f.endswith(".npy"):
                    t = int(f.split("_")[-1].split(".")[0])
                    rows.append({
                        "subject": subj.replace("subject_", "S"),
                        "emotion": EMOTION_MAP[TRIAL_LABELS[t - 1]],
                        "density": compute_density(
                            np.load(os.path.join(ssp, f))),
                    })
    df = pd.DataFrame(rows)
    print(f"  {len(df)} trial; densitas rata-rata {df.density.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor="white")
    order = ["negative", "neutral", "positive"]
    sns.boxplot(data=df, x="emotion", y="density", order=order, ax=axes[0],
                color="#4A90D9", showfliers=False, linewidth=2.5)

    # 672/675 trial punya densitas persis sama (0,10021); 3 trial pencilan
    # S01 (~0,93) memaksa skala default membentang sampai ~0,95 sehingga
    # box asli (672 trial) terjepit tak terlihat di dasar grafik. Zoom
    # sumbu-y ke rentang bulk data, lalu tandai pencilan lewat anotasi
    # teks alih-alih membiarkannya menentukan skala.
    bulk = df.loc[df["density"] < 0.3, "density"]
    outliers = df.loc[df["density"] >= 0.3, "density"]
    pad = max((bulk.max() - bulk.min()) * 2, 0.003)
    axes[0].set_ylim(bulk.min() - pad, bulk.max() + pad)
    if len(outliers):
        axes[0].text(
            0.03, 0.95,
            f"+{len(outliers)} trial pencilan (S01)\n"
            f"densitas $\\approx${outliers.mean():.2f} (di luar skala)",
            transform=axes[0].transAxes, fontsize=FS_TICK, ha="left",
            va="top", color="#B00000",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor="#B00000", alpha=0.9))

    axes[0].set_title("Densitas Graf GC per Kelas Emosi",
                      fontsize=FS_TITLE, fontweight="bold")
    axes[0].set_ylabel("Densitas jaringan", fontsize=FS_LABEL)
    axes[0].set_xlabel("Kelas emosi", fontsize=FS_LABEL)

    df.groupby("subject")["density"].mean().plot(
        kind="bar", ax=axes[1], color="#4A90D9")
    axes[1].set_title("Rata-rata Densitas GC per Subjek",
                      fontsize=FS_TITLE, fontweight="bold")
    axes[1].set_ylabel("Densitas jaringan", fontsize=FS_LABEL)
    axes[1].set_xlabel("Subjek", fontsize=FS_LABEL)
    axes[1].axhline(PROPORTIONAL_RETAIN, color="red", linestyle="--",
                    label=f"Target {PROPORTIONAL_RETAIN:.2f}")
    axes[1].legend(fontsize=FS_LABEL)
    for ax in axes:
        ax.tick_params(axis="both", labelsize=FS_TICK)
    plt.setp(axes[1].get_xticklabels(), rotation=90)
    plt.tight_layout()
    out = os.path.join(GC_FIG, "gc_density_analysis_optimized.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


def fig_subject15():
    """Tiga sesi subjek 15 -- subjek yang diproses lewat notebook terpisah."""
    sessions = [("session_20130709", "Sesi 1 (2013-07-09)"),
                ("session_20131016", "Sesi 2 (2013-10-16)"),
                ("session_20131105", "Sesi 3 (2013-11-05)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor="white")
    for ax, (sess, title) in zip(axes, sessions):
        m = np.load(os.path.join(GC_DIR, "subject_15", sess,
                                 "gc_thresholded_trial_01.npy"))
        sns.heatmap(m, ax=ax, cmap="hot", xticklabels=False,
                    yticklabels=False, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{title}\ndensitas={compute_density(m):.3f}",
                     fontsize=FS_TITLE, fontweight="bold")
        ax.set_xlabel("Kanal sumber", fontsize=FS_LABEL)
        ax.set_ylabel("Kanal target", fontsize=FS_LABEL)
        _cbar(ax)
    plt.suptitle("Subjek 15 — Matriks GC Hasil Ambang (Trial 1 per Sesi)",
                 fontsize=FS_SUPTITLE, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(GC_FIG, "subject_15_gc_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out)


if __name__ == "__main__":
    os.makedirs(GC_FIG, exist_ok=True)
    os.makedirs(PDC_FIG, exist_ok=True)
    fig_pdc_bands()
    fig_gc_sample()
    fig_gc_density()
    fig_subject15()
