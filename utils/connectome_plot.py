"""
Shared MNE-based directed-connectivity topomap plotter (GC & PDC).
=====================================================================
Menggantikan kepala/telinga/hidung yang digambar manual (lingkaran +
elips buatan tangan) dengan geometri kepala & posisi elektroda ASLI
dari montage MNE-Python (`mne.channels.read_custom_montage`, memakai
file `channel_62_pos.locs` yang sama dengan seluruh pipeline lain).

Perbaikan dibanding versi lama (`granger_causality_all_subjects_2d_viz.ipynb`,
`pdc_2d_topdown_viz.ipynb`):
  1. Posisi elektroda dari montage asli (bukan template standard_1005 yang
     di-scale paksa x12) -> semua elektroda, termasuk CB1/CB2, dijamin di
     dalam outline kepala (tidak ada panah yang keluar lingkaran).
  2. Outline kepala/hidung/telinga memakai fungsi internal MNE yang sama
     dipakai `mne.viz.plot_topomap` (`_get_pos_outlines`), bukan digambar
     manual.
  3. Label channel HANYA ditampilkan untuk elektroda yang punya minimal
     1 koneksi tersimpan (setelah threshold) -- mencegah 62 label bertumpuk
     di elektroda yang tidak relevan.
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

mne.set_log_level("ERROR")

CHANNEL_FILE = r"D:\Skripsi\dataset\SEED\SEED\channel_62_pos.locs"

CHANNEL_NAMES = [
    "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz",
    "O2", "CB2",
]

_FRONTAL = {"FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"}
_CENTRAL = {"FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"}
_PARIETAL = {"TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"}
_OCCIPITAL = {"PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"}

LOBE_COLORS = {"frontal": "#E84A5F", "central": "#3EC1D3", "parietal": "#F9A826", "occipital": "#4A47A3"}


def _lobe(ch_name):
    u = ch_name.upper()
    if u in _FRONTAL:
        return "frontal"
    if u in _CENTRAL:
        return "central"
    if u in _PARIETAL:
        return "parietal"
    if u in _OCCIPITAL:
        return "occipital"
    return "other"


NODE_COLORS = [LOBE_COLORS.get(_lobe(name), "#95A5A6") for name in CHANNEL_NAMES]


def cap_top_edges(adj_matrix, max_edges):
    """Keep only the `max_edges` strongest already-significant edges.

    Pure display thinning: the caller must threshold for significance
    FIRST (e.g. real p-value or global percentile cutoff) -- this only
    limits how many of those significant edges get drawn, so a dense
    graph stays legible. Does not change which edges count as
    "significant" for any downstream statistic.
    """
    work = adj_matrix.copy().astype(float)
    np.fill_diagonal(work, 0.0)
    n_valid = np.count_nonzero(work)
    if n_valid <= max_edges:
        return work
    thr = np.percentile(work[work > 0], 100 - (100 * max_edges / n_valid))
    return np.where(work >= thr, work, 0.0)


def build_montage_layout(channel_names=CHANNEL_NAMES):
    """Load real electrode positions + head/nose/ear outlines from MNE.

    Returns
    -------
    pos : (n_channels, 2) array -- 2D electrode coordinates used by MNE's
        own topomap renderer (guaranteed inside the head outline).
    outlines : dict with 'head', 'nose', 'ear_left', 'ear_right' polylines
        plus 'clip_radius' -- identical to what `mne.viz.plot_topomap` draws.
    """
    montage = mne.channels.read_custom_montage(CHANNEL_FILE)
    info = mne.create_info(channel_names, sfreq=100.0, ch_types="eeg")
    info.set_montage(montage)

    from mne.viz.topomap import _get_pos_outlines

    picks = list(range(len(channel_names)))
    pos, outlines = _get_pos_outlines(info, picks, sphere=None)
    return pos, outlines


def draw_head_outline(ax, outlines):
    for key in ("head", "nose", "ear_left", "ear_right"):
        x, y = outlines[key]
        ax.plot(x, y, color="#7F8C8D", linewidth=1.8, zorder=1)


def plot_directed_connectome(
    ax,
    adj_matrix,
    pos,
    outlines,
    channel_names,
    title,
    cmap_name="YlOrRd",
    node_colors=None,
    title_fontsize=15,
    label_fontsize=7.5,
    label_min_dist_factor=0.32,
):
    """Draw one directed-connectivity topomap on `ax`.

    `adj_matrix` must already be thresholded (zeros = no edge); this
    function does not do any additional filtering, so the caller controls
    exactly which edges count as "significant" (kept identical to
    upstream pipeline logic).
    """
    node_colors = node_colors or NODE_COLORS
    clip_r = outlines["clip_radius"][0]

    draw_head_outline(ax, outlines)

    work = adj_matrix.copy().astype(float)
    np.fill_diagonal(work, 0.0)

    active = work > 0
    degrees = active.sum(axis=0) + active.sum(axis=1)

    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", color="#1A252C", pad=12)

    if not np.any(active):
        for idx, name in enumerate(channel_names):
            x, y = pos[idx]
            ax.scatter(x, y, s=18, color=node_colors[idx], zorder=10, edgecolors="white", linewidths=0.6)
        ax.set_xlim(-clip_r * 1.35, clip_r * 1.35)
        ax.set_ylim(-clip_r * 1.35, clip_r * 1.55)
        ax.set_aspect("equal")
        ax.axis("off")
        return

    deg_min, deg_max = degrees.min(), degrees.max()
    node_sizes = 25 + 90 * (degrees - deg_min) / (deg_max - deg_min + 1e-9)

    connections = []
    n_nodes = len(channel_names)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if work[i, j] > 0:
                connections.append((i, j, work[i, j]))
    connections.sort(key=lambda c: c[2])

    cmap = plt.colormaps.get_cmap(cmap_name)
    strengths = [c[2] for c in connections]
    norm = Normalize(vmin=min(strengths), vmax=max(strengths))

    shrink = clip_r * 0.09
    for i, j, strength in connections:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        dx, dy = x2 - x1, y2 - y1
        dist = np.hypot(dx, dy)
        if dist == 0:
            continue
        x1n, y1n = x1 + shrink * dx / dist, y1 + shrink * dy / dist
        x2n, y2n = x2 - shrink * dx / dist, y2 - shrink * dy / dist

        color = cmap(norm(strength))
        linewidth = 0.8 + 3.2 * norm(strength)
        alpha = 0.65 + 0.35 * norm(strength)

        ax.annotate(
            "",
            xy=(x2n, y2n),
            xytext=(x1n, y1n),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                lw=linewidth,
                mutation_scale=9,
                connectionstyle="arc3,rad=0.15",
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )

    # Nodes: draw all (small, grey-ish if inactive) but only LABEL active ones.
    # Active electrodes on a 62-channel montage can sit very close together
    # (e.g. TP7/CP5, PO7/PO5, O1/CB1), so labelling every active node makes
    # neighbouring names collide. Greedily keep the label of the
    # higher-degree (more important) node and drop labels that would land
    # too close to one already placed -- the dot is still drawn either way.
    active_idx = [idx for idx in range(len(channel_names)) if degrees[idx] > 0]
    active_idx.sort(key=lambda i: degrees[i], reverse=True)
    # Faktor 0,32 dikalibrasi untuk panel tunggal GC (35 edge, satu kepala
    # besar). Panel PDC 3-kolom jauh lebih padat (40 edge, kepala lebih kecil)
    # sehingga butuh faktor lebih besar -- lewat `label_min_dist_factor`.
    # Menaikkan label_fontsize saja tidak menolong: lebar teks ikut naik
    # linier, jadi rasionya tetap.
    min_label_dist = clip_r * label_min_dist_factor * (label_fontsize / 23.0)
    placed_positions = []
    labelled = set()
    for idx in active_idx:
        x, y = pos[idx]
        if all(np.hypot(x - px, y - py) >= min_label_dist for px, py in placed_positions):
            placed_positions.append((x, y))
            labelled.add(idx)

    for idx, name in enumerate(channel_names):
        x, y = pos[idx]
        is_active = degrees[idx] > 0
        color = node_colors[idx] if is_active else "#D5DBDB"
        size = node_sizes[idx] if is_active else 10
        ax.scatter(x, y, s=size, color=color, zorder=10 if is_active else 4,
                   edgecolors="white" if is_active else "none", linewidths=1.0, alpha=0.95 if is_active else 0.5)
        if idx in labelled:
            ax.text(x, y + clip_r * 0.09, name, ha="center", va="bottom",
                    fontsize=label_fontsize, fontweight="bold", color="#2C3E50", zorder=12)

    ax.set_xlim(-clip_r * 1.35, clip_r * 1.35)
    ax.set_ylim(-clip_r * 1.35, clip_r * 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
