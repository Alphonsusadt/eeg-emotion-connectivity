"""
Partial Directed Coherence (PDC) Analyzer
===========================================
Compute PDC from MVAR model per trial, 5 frequency bands.
"""

import numpy as np
import os
import sys
import warnings
from statsmodels.tsa.api import VAR

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PDC_MAX_ORDER, PDC_MIN_ORDER, PDC_FREQUENCY_BANDS, TARGET_FS
)


def fit_mvar(data, max_order=None, min_order=None):
    """
    Fit MVAR model and select optimal order via AIC.

    Args:
        data: (n_timepoints, n_channels)
        max_order: Maximum model order
        min_order: Minimum model order

    Returns:
        model_fitted: Fitted VAR model
        optimal_order: Selected order
        coefficients: (order, n_channels, n_channels)
    """
    max_order = max_order or PDC_MAX_ORDER
    min_order = min_order or PDC_MIN_ORDER

    model = VAR(data)

    # Select order
    try:
        order_result = model.select_order(maxlags=max_order)
        optimal_order = order_result.aic
        optimal_order = max(optimal_order, min_order)
    except:
        optimal_order = 5

    model_fitted = model.fit(maxlags=optimal_order)

    n_channels = data.shape[1]
    coefficients = np.zeros((optimal_order, n_channels, n_channels))

    # FIX: statsmodels VAR.fit() prepends deterministic terms (const, trend, ...)
    # to `params` before the per-lag coefficient blocks (k_trend rows for the
    # default trend='c'). The old code started reading blocks at row `lag*n_channels`
    # with no offset, so every extracted A_lag matrix was shifted by one row
    # (mixing in the intercept / neighboring lag's row) and the true last row of
    # the highest-lag block was silently dropped. Skip the k_trend rows first.
    n_trend = model_fitted.k_trend
    for lag in range(optimal_order):
        start = n_trend + lag * n_channels
        end = start + n_channels
        if end <= model_fitted.params.shape[0]:
            coefficients[lag] = model_fitted.params[start:end].T

    return model_fitted, optimal_order, coefficients


def compute_pdc(coefficients, freqs, fs):
    """
    Compute PDC from MVAR coefficients.

    PDC_ij(f) = |A_ij(f)| / sqrt(sum_m |A_mj(f)|^2)

    Args:
        coefficients: (order, n_channels, n_channels)
        freqs: Array of frequencies to evaluate
        fs: Sampling frequency

    Returns:
        pdc: (n_freqs, n_channels, n_channels)
    """
    order, n_ch, _ = coefficients.shape
    n_freqs = len(freqs)
    pdc = np.zeros((n_freqs, n_ch, n_ch))

    for f_idx, freq in enumerate(freqs):
        # A(f) = I - sum_k A_k * exp(-2*pi*i*f*k/fs)
        A_f = np.eye(n_ch, dtype=complex)
        for lag in range(1, order + 1):
            exp_term = np.exp(-2j * np.pi * freq * lag / fs)
            if lag - 1 < coefficients.shape[0]:
                A_f -= coefficients[lag - 1] * exp_term

        # PDC: |A_ij(f)| / sqrt(sum_m |A_mj(f)|^2)
        for j in range(n_ch):
            denom = np.sqrt(np.sum(np.abs(A_f[:, j]) ** 2))
            if denom > 0:
                for i in range(n_ch):
                    pdc[f_idx, i, j] = np.abs(A_f[i, j]) / denom

    return pdc


def average_over_bands(pdc, freqs, bands=None):
    """
    Average PDC over frequency bands.

    Args:
        pdc: (n_freqs, n_channels, n_channels)
        freqs: Frequency values
        bands: Dict {band_name: (fmin, fmax)}

    Returns:
        band_pdc: Dict {band_name: (n_channels, n_channels)}
    """
    bands = bands or PDC_FREQUENCY_BANDS
    band_pdc = {}

    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if np.sum(mask) > 0:
            band_pdc[band_name] = np.mean(pdc[mask], axis=0)
        else:
            band_pdc[band_name] = np.zeros((pdc.shape[1], pdc.shape[2]))

    # Broadband average
    band_pdc['broadband'] = np.mean(pdc, axis=0)

    return band_pdc


def analyze_trial(eeg_data, fs=None, max_order=None, bands=None):
    """
    Full PDC analysis for one trial.

    Args:
        eeg_data: (n_channels, n_timepoints) preprocessed EEG
        fs: Sampling frequency
        max_order: Max MVAR order
        bands: Frequency bands dict

    Returns:
        band_pdc: Dict {band_name: (n_channels, n_channels)}
        optimal_order: Selected MVAR order
    """
    fs = fs or TARGET_FS
    max_order = max_order or PDC_MAX_ORDER
    bands = bands or PDC_FREQUENCY_BANDS

    # Transpose for MVAR: (timepoints, channels)
    data = eeg_data.T

    # Fit MVAR
    _, optimal_order, coefficients = fit_mvar(data, max_order)

    # Define frequencies
    freqs = np.linspace(0.5, fs / 2, 200)

    # Compute PDC
    pdc_full = compute_pdc(coefficients, freqs, fs)

    # Average over bands
    band_pdc = average_over_bands(pdc_full, freqs, bands)

    return band_pdc, optimal_order
