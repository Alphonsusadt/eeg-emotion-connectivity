"""
Regenerasi `gc_2d_topdown_*.png` memakai geometri kepala/elektroda MNE asli
(lihat `new_data/utils/connectome_plot.py`) supaya tidak ada label yang
bertumpuk dan tidak ada panah yang keluar dari outline kepala.

Sumber data & logika threshold SENGAJA TIDAK DIUBAH (permintaan user):
tetap folder legacy `D:\\Skripsi\\HasilGrangerPDC\\GrangerCausality\\`,
filter signifikansi p < 0.01, lalu ambil `n_lines` koneksi terkuat --
persis logika `granger_causality_all_subjects_2d_viz.ipynb`. Hanya
RENDERING yang diganti.

Usage:
    python regen_gc_topdown_mne.py --subjects 1          # sampel 1 subjek
    python regen_gc_topdown_mne.py --subjects all        # semua subjek (batch penuh)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils"))
from connectome_plot import CHANNEL_NAMES, build_montage_layout, plot_directed_connectome  # noqa: E402

LEGACY_GC_DIR = r"D:\Skripsi\HasilGrangerPDC\GrangerCausality"
OUTPUT_BASE_DIR = r"D:\Skripsi\new_data\01_granger_causality\figures"
EMOTIONS = ["negative", "neutral", "positive"]
N_LINES = 35
P_ALPHA = 0.01


def threshold_gc(gc_matrix, p_matrix, n_lines=N_LINES, p_alpha=P_ALPHA):
    work = gc_matrix.copy()
    np.fill_diagonal(work, 0)
    significant = p_matrix < p_alpha
    filtered = work * significant
    if np.any(filtered > 0):
        total_valid = np.count_nonzero(filtered)
        if total_valid > n_lines:
            thr = np.percentile(filtered[filtered > 0], 100 - (100 * n_lines / total_valid))
        else:
            thr = filtered[filtered > 0].min()
    else:
        return np.zeros_like(work)
    return np.where(filtered >= thr, filtered, 0)


def process_session(subject, session, pos, outlines):
    session_dir = os.path.join(LEGACY_GC_DIR, subject, session)
    if not all(os.path.exists(os.path.join(session_dir, f"gc_matrix_{e}.npy")) for e in EMOTIONS):
        return False

    out_dir = os.path.join(OUTPUT_BASE_DIR, subject, session)
    os.makedirs(out_dir, exist_ok=True)

    adj_by_emotion = {}
    for emotion in EMOTIONS:
        gc_matrix = np.load(os.path.join(session_dir, f"gc_matrix_{emotion}.npy"))
        p_matrix = np.load(os.path.join(session_dir, f"p_matrix_{emotion}.npy"))
        adj_by_emotion[emotion] = threshold_gc(gc_matrix, p_matrix)

    # Comparison (3 panel side-by-side)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), facecolor="white")
    for ax, emotion in zip(axes, EMOTIONS):
        plot_directed_connectome(ax, adj_by_emotion[emotion], pos, outlines, CHANNEL_NAMES,
                                  f"{emotion.upper()} Emotion", cmap_name="YlOrRd")
    plt.suptitle(f"EEG Granger Causality Connectivity Maps (Top-Down View)\n"
                 f"{subject.replace('_', ' ').title()} | Session: {session}",
                 fontsize=18, fontweight="bold", color="#1A252C", y=1.03)
    plt.savefig(os.path.join(out_dir, "gc_2d_topdown_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Individual per emotion.
    # These files are placed 2-per-row at 0.48\textwidth in the thesis (~2.82in
    # print width vs this 7.5in canvas), so title/label fonts are bumped up
    # front so text stays legible once LaTeX shrinks the image.
    for emotion in EMOTIONS:
        fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
        title = f"Granger Causality - {emotion.upper()}\n{subject.replace('_', ' ').title()} | Session: {session}"
        plot_directed_connectome(ax, adj_by_emotion[emotion], pos, outlines, CHANNEL_NAMES,
                                  title, cmap_name="YlOrRd", title_fontsize=29, label_fontsize=23)
        plt.savefig(os.path.join(out_dir, f"gc_2d_topdown_{emotion}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="1", help="'1' style number(s) comma-separated, or 'all'")
    args = parser.parse_args()

    pos, outlines = build_montage_layout()

    all_subjects = sorted(
        d for d in os.listdir(LEGACY_GC_DIR)
        if os.path.isdir(os.path.join(LEGACY_GC_DIR, d)) and d.startswith("subject_")
    )
    if args.subjects == "all":
        subjects = all_subjects
    else:
        wanted = {f"subject_{s.strip()}" for s in args.subjects.split(",")}
        subjects = [s for s in all_subjects if s in wanted]

    print(f"Processing {len(subjects)} subject(s): {subjects}")
    processed = 0
    for subject in tqdm(subjects, desc="Subjects"):
        sub_path = os.path.join(LEGACY_GC_DIR, subject)
        sessions = sorted(
            d for d in os.listdir(sub_path)
            if os.path.isdir(os.path.join(sub_path, d)) and d.isdigit()
        )
        for session in sessions:
            if process_session(subject, session, pos, outlines):
                processed += 1

    print(f"Done. {processed} session(s) regenerated -> {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
