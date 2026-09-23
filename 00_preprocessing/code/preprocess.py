"""
EEG Preprocessing
==================
Bandpass filter + notch filter + downsample untuk SEED EEG data.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly
import os
import sys
import json
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    ORIGINAL_FS, TARGET_FS, BANDPASS_LOW, BANDPASS_HIGH,
    BANDPASS_ORDER, NOTCH_FREQ, NOTCH_Q, PREPROCESS_DIR
)


def bandpass_filter(data, low, high, fs, order=4):
    """
    Apply bandpass filter to EEG data.

    Args:
        data: (n_channels, n_timepoints)
        low: Low cutoff frequency (Hz)
        high: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order

    Returns:
        Filtered data (n_channels, n_timepoints)
    """
    sos = butter(order, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, data, axis=1)


def notch_filter(data, freq, q, fs):
    """
    Apply notch filter to remove line noise.

    Args:
        data: (n_channels, n_timepoints)
        freq: Notch frequency (Hz)
        q: Quality factor
        fs: Sampling frequency (Hz)

    Returns:
        Filtered data (n_channels, n_timepoints)
    """
    b, a = iirnotch(freq, q, fs)
    return filtfilt(b, a, data, axis=1)


def downsample(data, original_fs, target_fs):
    """
    Downsample EEG data using resample_poly (anti-aliasing).

    Args:
        data: (n_channels, n_timepoints)
        original_fs: Original sampling rate
        target_fs: Target sampling rate

    Returns:
        Downsampled data (n_channels, new_timepoints)
    """
    # Compute GCD for up/down factors
    from math import gcd
    g = gcd(original_fs, target_fs)
    up = target_fs // g
    down = original_fs // g
    return resample_poly(data, up, down, axis=1)


def preprocess_eeg(eeg_data, original_fs=None, target_fs=None,
                   bp_low=None, bp_high=None, bp_order=None,
                   notch_freq=None, notch_q=None):
    """
    Full preprocessing pipeline for a single trial.

    Args:
        eeg_data: (n_channels, n_timepoints) raw EEG

    Returns:
        Preprocessed EEG data (n_channels, new_timepoints)
    """
    original_fs = original_fs or ORIGINAL_FS
    target_fs = target_fs or TARGET_FS
    bp_low = bp_low or BANDPASS_LOW
    bp_high = bp_high or BANDPASS_HIGH
    bp_order = bp_order or BANDPASS_ORDER
    notch_freq = notch_freq or NOTCH_FREQ
    notch_q = notch_q or NOTCH_Q

    # Step 1: Bandpass filter
    filtered = bandpass_filter(eeg_data, bp_low, bp_high, original_fs, bp_order)

    # Step 2: Notch filter (only if notch_freq < original_fs/2)
    if notch_freq < original_fs / 2:
        filtered = notch_filter(filtered, notch_freq, notch_q, original_fs)

    # Step 3: Downsample
    downsampled = downsample(filtered, original_fs, target_fs)

    return downsampled


def save_preprocessed(eeg_data, subject_id, session_date, trial_idx, output_dir):
    """Save preprocessed trial to .npy file."""
    subj_dir = os.path.join(output_dir, f"subject_{subject_id:02d}")
    sess_dir = os.path.join(subj_dir, f"session_{session_date}")
    os.makedirs(sess_dir, exist_ok=True)

    filepath = os.path.join(sess_dir, f"trial_{trial_idx+1:02d}.npy")
    np.save(filepath, eeg_data)
    return filepath


if __name__ == "__main__":
    # Test preprocessing on one trial
    from seed_loader import SEEDLoader

    loader = SEEDLoader()
    eeg, label = loader.load_trial_eeg(1, '20131027', 0)

    if eeg is not None:
        print(f"Raw EEG: shape={eeg.shape}, fs={ORIGINAL_FS}Hz")
        print(f"  Duration: {eeg.shape[1]/ORIGINAL_FS:.1f}s")

        processed = preprocess_eeg(eeg)
        print(f"Preprocessed: shape={processed.shape}, fs={TARGET_FS}Hz")
        print(f"  Duration: {processed.shape[1]/TARGET_FS:.1f}s")
        print(f"  Label: {label}")
