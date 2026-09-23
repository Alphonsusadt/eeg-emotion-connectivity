"""Montase 15 subjek (satu panel per subjek) untuk Bab 4 subbab 4.1.6.

Panduan Bab 4 "Lapis 2": ratusan topomap per-subjek tidak mungkin ditempel
satu per satu, tetapi juga tidak boleh dibiarkan tak dirujuk. Solusinya satu
gambar berisi 15 panel dengan kondisi difiksasi (emosi positif, sesi pertama)
sehingga pembaca langsung melihat subjek mana yang menyimpang -- persis
argumen 4.1.6.

Dua keluaran:
  gc_montage_15_subjects.png   -- GC, dari matriks legacy + p-matrix (p < 0,01)
  pdc_montage_15_subjects.png  -- PDC broadband, dari matriks TERKOREKSI
                                  (output/pdc_matrices/, hasil perbaikan
                                  20 Juli 2026), bukan dari PNG lama yang
                                  dirender sebelum perbaikan.

Logika threshold sengaja identik dengan regen_gc_topdown_mne.py dan
regen_pdc_topdown_mne.py supaya montase konsisten dengan topomap in-text.

Usage:
    python make_subject_montage.py
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from connectome_plot import (  # noqa: E402
    CHANNEL_NAMES, LOBE_COLORS, build_montage_layout, plot_directed_connectome,
    cap_top_edges,
)

LEGACY_GC_DIR = r"D:\Skripsi\HasilGrangerPDC\GrangerCausality"
PDC_DIR = r"D:\Skripsi\new_data\02_pdc\output\pdc_matrices"
PDC_CUTOFFS = r"D:\Skripsi\new_data\02_pdc\code\pdc_global_cutoffs.json"
OUT_DIR = r"D:\Skripsi\new_data\phase_2_statistics\2.1_individual_analysis\figures"

EMOTION = "positive"          # kelas yang memicu efek signifikan di 4.1.2
PDC_BAND = "broadband"        # pita PDC terdekat ke ambang (p = 0,093)
N_LINES_GC = 35
P_ALPHA_GC = 0.01
MAX_EDGES_PDC = 40
TRIAL_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
POSITIVE_TRIALS = [i + 1 for i, lb in enumerate(TRIAL_LABELS) if lb == 1]

# Kanvas 20x20 in dicetak pada \textwidth ~ 5,87 in (skala ~0,29), jadi
# seluruh font dinaikkan ~3,4x dari ukuran "wajar" matplotlib.
GRID = (4, 4)
FIGSIZE = (20, 20)
PANEL_TITLE_FS = 34
SUPTITLE_FS = 40
LEGEND_FS = 26


def threshold_gc(gc_matrix, p_matrix, n_lines=N_LINES_GC, p_alpha=P_ALPHA_GC):
    work = gc_matrix.copy()
    np.fill_diagonal(work, 0)
    filtered = work * (p_matrix < p_alpha)
    if not np.any(filtered > 0):
        return np.zeros_like(work)
    total_valid = np.count_nonzero(filtered)
    if total_valid > n_lines:
        thr = np.percentile(filtered[filtered > 0],
                            100 - (100 * n_lines / total_valid))
    else:
        thr = filtered[filtered > 0].min()
    return np.where(filtered >= thr, filtered, 0)


def threshold_pdc(matrix, cutoff):
    m = matrix.copy().astype(float)
    np.fill_diagonal(m, 0.0)
    m[m < cutoff] = 0.0
    return cap_top_edges(m, MAX_EDGES_PDC)


def first_session(path, prefix=''):
    subs = sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d))
                  and d.startswith(prefix) and d != 'figures')
    return subs[0] if subs else None


def collect_gc():
    """-> list of (label, adj_matrix) untuk 15 subjek."""
    out = []
    for n in range(1, 16):
        subj = f'subject_{n}'
        sub_path = os.path.join(LEGACY_GC_DIR, subj)
        sess = first_session(sub_path)
        d = os.path.join(sub_path, sess)
        gc = np.load(os.path.join(d, f'gc_matrix_{EMOTION}.npy'))
        pm = np.load(os.path.join(d, f'p_matrix_{EMOTION}.npy'))
        out.append((f'S{n:02d}', threshold_gc(gc, pm)))
        print(f'  GC  {subj}/{sess}: {np.count_nonzero(out[-1][1])} edge')
    return out


def collect_pdc():
    with open(PDC_CUTOFFS) as f:
        cutoff = json.load(f)[PDC_BAND]
    out = []
    for n in range(1, 16):
        subj = f'subject_{n:02d}'
        sub_path = os.path.join(PDC_DIR, subj)
        sess = first_session(sub_path, 'session_')
        d = os.path.join(sub_path, sess)
        mats = [np.load(os.path.join(d, f'pdc_{PDC_BAND}_trial_{t:02d}.npy'))
                for t in POSITIVE_TRIALS
                if os.path.exists(os.path.join(
                    d, f'pdc_{PDC_BAND}_trial_{t:02d}.npy'))]
        avg = np.mean(mats, axis=0)
        out.append((f'S{n:02d}', threshold_pdc(avg, cutoff)))
        print(f'  PDC {subj}/{sess}: {len(mats)} trial, '
              f'{np.count_nonzero(out[-1][1])} edge')
    return out, cutoff


def render(panels, title, footnote, cmap, out_name):
    pos, outlines = build_montage_layout()
    rows, cols = GRID
    fig, axes = plt.subplots(rows, cols, figsize=FIGSIZE, facecolor='white')
    flat = axes.ravel()

    for ax, (label, adj) in zip(flat, panels):
        # label_fontsize=0 -> nama kanal tidak digambar; pada montase yang
        # dibaca adalah POLA sebarannya, bukan nama elektrodanya.
        plot_directed_connectome(ax, adj, pos, outlines, CHANNEL_NAMES,
                                 label, cmap_name=cmap,
                                 title_fontsize=PANEL_TITLE_FS,
                                 label_fontsize=0)

    # Sel ke-16 dipakai sebagai legenda lobus, bukan dibiarkan kosong.
    legend_ax = flat[len(panels)]
    legend_ax.axis('off')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=22,
                      markerfacecolor=c, markeredgecolor='white', label=k.title())
               for k, c in LOBE_COLORS.items()]
    legend_ax.legend(handles=handles, loc='center', frameon=False,
                     fontsize=LEGEND_FS, title='Lobus', title_fontsize=LEGEND_FS,
                     labelspacing=1.1)
    for ax in flat[len(panels) + 1:]:
        ax.axis('off')

    # top=0.88 memberi ruang agar suptitle tidak menabrak judul panel baris
    # pertama; hspace kecil karena tiap panel sudah beraspek sama sehingga
    # menyisakan ruang kosongnya sendiri.
    fig.suptitle(title, fontsize=SUPTITLE_FS, fontweight='bold',
                 color='#1A252C', y=0.955)
    fig.text(0.5, 0.045, footnote, ha='center', fontsize=LEGEND_FS,
             color='#555555', wrap=True)
    fig.subplots_adjust(top=0.845, bottom=0.08, wspace=0.02, hspace=0.12)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, out_name)
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    print('Saved:', path)
    return path


if __name__ == '__main__':
    print('Mengumpulkan GC...')
    gc_panels = collect_gc()
    render(
        gc_panels,
        'Peta Konektivitas Efektif GC per Subjek\nKondisi Emosi Positif',
        f'Sesi pertama tiap subjek; edge signifikan p < {P_ALPHA_GC} '
        f'dan {N_LINES_GC} koneksi terkuat ditampilkan. '
        'Nama kanal sengaja tidak dicetak agar pola antar-subjek dapat dibandingkan.',
        'YlOrRd', 'gc_montage_15_subjects.png')

    print('Mengumpulkan PDC...')
    pdc_panels, cutoff = collect_pdc()
    render(
        pdc_panels,
        f'Peta Konektivitas Efektif PDC Pita {PDC_BAND.title()} per Subjek\n'
        'Kondisi Emosi Positif',
        f'Sesi pertama tiap subjek, dihitung dari matriks PDC terkoreksi; '
        f'edge signifikan PDC >= cutoff persentil-75 global ({cutoff:.4f}), '
        f'maks. {MAX_EDGES_PDC} edge terkuat per panel.',
        'Blues', 'pdc_montage_15_subjects.png')
