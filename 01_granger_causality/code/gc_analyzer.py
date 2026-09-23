"""
Granger Causality Analyzer
===========================
Compute pairwise GC with FDR threshold per trial.
"""

import numpy as np
import os
import sys
import warnings
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GC_MAX_LAG, GC_ALPHA, GC_THRESHOLD_METHOD, GC_PERCENTILE, N_CHANNELS


def check_stationarity(ts):
    """Quick stationarity check using ADF test."""
    from statsmodels.tsa.stattools import adfuller
    try:
        result = adfuller(ts, autolag='AIC')
        return result[1] < 0.05
    except:
        return False


def make_stationary(data):
    """Apply differencing to non-stationary channels."""
    n_timepoints, n_channels = data.shape
    result = np.zeros_like(data)
    for ch in range(n_channels):
        ts = data[:, ch]
        if not check_stationarity(ts):
            ts = np.diff(ts)
            result[:len(ts), ch] = ts
            result[len(ts):, ch] = ts[-1]  # pad last value
        else:
            result[:, ch] = ts
    return result


def normalize_channels(data):
    """Z-score normalize each channel."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std[std < 1e-10] = 1.0
    return (data - mean) / std


def preprocess_for_gc(eeg_data):
    """
    Preprocess EEG for Granger Causality.

    Args:
        eeg_data: (n_channels, n_timepoints)

    Returns:
        data: (n_timepoints, n_channels) stationary & normalized
    """
    data = eeg_data.T  # (time, channels)
    data = normalize_channels(data)
    data = make_stationary(data)
    return data


def compute_gc_pairwise(data, max_lag=None):
    """
    Compute pairwise Granger Causality F-statistics.

    Args:
        data: (n_timepoints, n_channels) preprocessed
        max_lag: Maximum lag to test

    Returns:
        gc_matrix: (n_channels, n_channels) F-statistic values
        p_matrix: (n_channels, n_channels) p-values (min across lags)
    """
    max_lag = max_lag or GC_MAX_LAG
    n_ch = data.shape[1]
    gc_matrix = np.zeros((n_ch, n_ch))
    p_matrix = np.ones((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(n_ch):
            if i == j:
                continue
            try:
                test_data = data[:, [j, i]]  # grangercausalitytests(data[:,[a,b]]) tests
                                              # whether b causes a -> this tests i causes j
                                              # (stored at gc_matrix[i, j]: row i = source)
                if np.any(np.isnan(test_data)) or np.any(np.isinf(test_data)):
                    continue

                result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # Best (minimum p-value) across lags
                best_p = 1.0
                best_f = 0.0
                for lag in range(1, max_lag + 1):
                    f_stat = result[lag][0]['ssr_ftest'][0]
                    p_val = result[lag][0]['ssr_ftest'][1]
                    if p_val < best_p:
                        best_p = p_val
                        best_f = f_stat

                gc_matrix[i, j] = best_f
                p_matrix[i, j] = best_p
            except:
                continue

    return gc_matrix, p_matrix


def apply_threshold(gc_matrix, p_matrix, method=None, alpha=None, percentile=None):
    """
    Apply threshold to GC matrix.

    Args:
        gc_matrix: F-statistic matrix
        p_matrix: P-value matrix
        method: "fdr", "bonferroni", or "percentile"
        alpha: Significance level
        percentile: Percentile threshold (0-100)

    Returns:
        thresholded: Thresholded GC matrix (non-significant = 0)
    """
    method = method or GC_THRESHOLD_METHOD
    alpha = alpha or GC_ALPHA
    percentile = percentile or GC_PERCENTILE

    n_ch = gc_matrix.shape[0]
    thresholded = gc_matrix.copy()

    # Always zero diagonal
    np.fill_diagonal(thresholded, 0)

    if method == "bonferroni":
        n_tests = n_ch * (n_ch - 1)
        alpha_corrected = alpha / n_tests
        mask = p_matrix >= alpha_corrected
        np.fill_diagonal(mask, True)
        thresholded[mask] = 0

    elif method == "fdr":
        # Flatten off-diagonal p-values
        mask_offdiag = ~np.eye(n_ch, dtype=bool)
        p_flat = p_matrix[mask_offdiag]

        reject, p_corrected, _, _ = multipletests(p_flat, alpha=alpha, method='fdr_bh')

        # Build reject matrix
        reject_matrix = np.ones((n_ch, n_ch), dtype=bool)
        reject_matrix[mask_offdiag] = reject
        np.fill_diagonal(reject_matrix, False)

        thresholded[~reject_matrix] = 0

    elif method == "percentile":
        nonzero = gc_matrix[gc_matrix > 0]
        if len(nonzero) > 0:
            threshold = np.percentile(nonzero, percentile)
            thresholded[thresholded < threshold] = 0
        # Also apply p-value threshold
        mask = p_matrix >= alpha
        np.fill_diagonal(mask, True)
        thresholded[mask] = 0

    return thresholded


def analyze_trial(eeg_data, max_lag=None, method=None):
    """
    Full GC analysis for one trial.

    Args:
        eeg_data: (n_channels, n_timepoints) preprocessed EEG
        max_lag: Max lag
        method: Threshold method

    Returns:
        gc_raw: Raw F-statistic matrix
        p_matrix: P-value matrix
        gc_thresholded: Thresholded matrix
        density: Network density after thresholding
    """
    data = preprocess_for_gc(eeg_data)
    gc_raw, p_matrix = compute_gc_pairwise(data, max_lag)
    gc_thresholded = apply_threshold(gc_raw, p_matrix, method)

    n_ch = gc_thresholded.shape[0]
    n_possible = n_ch * (n_ch - 1)
    n_edges = np.count_nonzero(gc_thresholded)
    density = n_edges / n_possible if n_possible > 0 else 0

    return gc_raw, p_matrix, gc_thresholded, density
