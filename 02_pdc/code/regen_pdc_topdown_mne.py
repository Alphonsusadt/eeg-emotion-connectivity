"""
Regenerasi `pdc_2d_topdown_*.png` memakai geometri kepala/elektroda MNE asli
(lihat `new_data/utils/connectome_plot.py`) DAN threshold signifikansi yang
sesungguhnya dipakai pipeline (`phase_1_data_structuring.ipynb`): cutoff
absolut per band, persentil ke-75 dipool dari SELURUH trial/subjek/sesi
band tsb (lihat `compute_pdc_cutoffs.py` -> `pdc_global_cutoffs.json`) --
BUKAN top-35 koneksi per panel seperti notebook lama (`pdc_2d_topdown_viz.ipynb`).

Rata-rata per emosi memakai pemetaan trial SEED standar yang sama dengan
notebook lama (TRIAL_LABELS).

Usage:
    python regen_pdc_topdown_mne.py --subjects 01          # sampel 1 subjek
    python regen_pdc_topdown_mne.py --subjects all         # semua subjek (batch penuh)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from connectome_plot import CHANNEL_NAMES, build_montage_layout, plot_directed_connectome, cap_top_edges  # noqa: E402

PDC_DIR = r"D:\Skripsi\new_data\02_pdc\output\pdc_matrices"
OUTPUT_BASE_DIR = r"D:\Skripsi\new_data\02_pdc\figures"
CUTOFFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdc_global_cutoffs.json")

PDC_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
MAX_DISPLAY_EDGES = 40  # legibility cap on top of the real significance cutoff (see cap_top_edges)
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]  # 1=positive 0=neutral -1=negative
EMOTION_TRIALS = {
    "positive": [i + 1 for i, lb in enumerate(TRIAL_LABELS) if lb == 1],
    "neutral": [i + 1 for i, lb in enumerate(TRIAL_LABELS) if lb == 0],
    "negative": [i + 1 for i, lb in enumerate(TRIAL_LABELS) if lb == -1],
}
EMOTIONS = ["negative", "neutral", "positive"]


def threshold_pdc(matrix, cutoff):
    """Real significance step: matches phase_1_data_structuring.ipynb exactly."""
    m = matrix.copy().astype(float)
    np.fill_diagonal(m, 0.0)
    m[m < cutoff] = 0.0
    return cap_top_edges(m, MAX_DISPLAY_EDGES)


def process_session(subject, session, band, cutoff, pos, outlines):
    session_dir = os.path.join(PDC_DIR, subject, session)
    trial_one = os.path.join(session_dir, f"pdc_{band}_trial_01.npy")
    if not os.path.exists(trial_one):
        return False

    emotion_matrices = {}
    for emotion in EMOTIONS:
        mats = []
        for t in EMOTION_TRIALS[emotion]:
            fp = os.path.join(session_dir, f"pdc_{band}_trial_{t:02d}.npy")
            if os.path.exists(fp):
                mats.append(np.load(fp))
        if mats:
            emotion_matrices[emotion] = np.mean(mats, axis=0)
    if len(emotion_matrices) < 3:
        return False

    out_dir = os.path.join(OUTPUT_BASE_DIR, subject, session)
    os.makedirs(out_dir, exist_ok=True)

    adj_by_emotion = {e: threshold_pdc(m, cutoff) for e, m in emotion_matrices.items()}

    # Berkas 3-panel ini dipasang pada 0,98\textwidth (~5,75 in) dan DITUMPUK
    # bertiga dalam satu halaman, jadi tinggi cetaknya hanya ~2 in per berkas.
    # Kanvas sengaja dijaga tetap ringkas (14 in, bukan 20 in) supaya faktor
    # penyusutan hanya ~0,41 dan font tidak perlu dinaikkan ekstrem. Percobaan
    # sebelumnya memakai kanvas 20 in + label_fontsize 34; hasilnya label kanal
    # justru menjadi terlalu besar RELATIF terhadap lingkaran kepala sehingga
    # nama kanal bertetangga saling menimpa ("T7C3", "CP2TP8"). label_fontsize
    # dikembalikan ke rentang terkalibrasi fungsi penekan label di
    # connectome_plot.py (min_label_dist dikalibrasi pada 23).
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), facecolor="white")
    for ax, emotion in zip(axes, EMOTIONS):
        # Just the emotion name (no "Emotion" suffix) so 3 side-by-side
        # titles at this font size don't run into each other.
        plot_directed_connectome(ax, adj_by_emotion[emotion], pos, outlines, CHANNEL_NAMES,
                                  emotion.upper(), cmap_name="Blues",
                                  title_fontsize=26, label_fontsize=23,
                                  label_min_dist_factor=0.60)
    # wspace kecil supaya lingkaran kepala mengisi lebar kanvas semaksimal
    # mungkin -- inilah sumber utama keterbacaan pada ukuran cetak.
    fig.subplots_adjust(wspace=0.02, top=0.82, bottom=0.10)
    # Judul dijadikan SATU baris; versi dua baris menabrak judul panel
    # (NEGATIVE/NEUTRAL/POSITIVE) pada kanvas seringkas ini.
    plt.suptitle(f"PDC {band.upper()} Band — "
                 f"{subject.replace('_', ' ').title()} | "
                 f"{session.replace('session_', '')}",
                 fontsize=25, fontweight="bold", color="#1A252C", y=0.98)
    fig.text(0.5, 0.02,
              f"Edge signifikan: PDC ≥ cutoff persentil-75 global band {band} ({cutoff:.4f}); "
              f"maks. {MAX_DISPLAY_EDGES} edge terkuat per panel.",
              ha="center", fontsize=17, color="#555555")
    plt.savefig(os.path.join(out_dir, f"pdc_2d_topdown_comparison_{band}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    for emotion in EMOTIONS:
        fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
        title = f"PDC {band.upper()} - {emotion.upper()}\n{subject.replace('_', ' ').title()} | {session.replace('_', ' ').title()}"
        plot_directed_connectome(ax, adj_by_emotion[emotion], pos, outlines, CHANNEL_NAMES,
                                  title, cmap_name="Blues", title_fontsize=29, label_fontsize=23)
        plt.savefig(os.path.join(out_dir, f"pdc_2d_topdown_{band}_{emotion}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="01", help="'01' style number(s) comma-separated, or 'all'")
    args = parser.parse_args()

    with open(CUTOFFS_PATH) as f:
        cutoffs = json.load(f)

    pos, outlines = build_montage_layout()

    all_subjects = sorted(
        d for d in os.listdir(PDC_DIR)
        if os.path.isdir(os.path.join(PDC_DIR, d)) and d.startswith("subject_")
    )
    if args.subjects == "all":
        subjects = all_subjects
    else:
        wanted = {f"subject_{s.strip().zfill(2)}" for s in args.subjects.split(",")}
        subjects = [s for s in all_subjects if s in wanted]

    print(f"Processing {len(subjects)} subject(s): {subjects}")
    processed = 0
    for subject in tqdm(subjects, desc="Subjects"):
        sub_path = os.path.join(PDC_DIR, subject)
        sessions = sorted(
            d for d in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, d)) and d.startswith("session_")
        )
        for session in sessions:
            for band in PDC_BANDS:
                if process_session(subject, session, band, cutoffs[band], pos, outlines):
                    processed += 1

    print(f"Done. {processed} session-band combo(s) regenerated -> {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
