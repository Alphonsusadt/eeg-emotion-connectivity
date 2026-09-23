"""
SEED Dataset Loader
====================
Load raw EEG data from SEED dataset.
Handles subject-session mapping, trial extraction, and labels.
"""

import scipy.io
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    DATASET_PATH, RAW_EEG_PATH, CHANNEL_FILE,
    SUBJECT_SESSIONS, TRIAL_LABELS, N_CHANNELS,
    EMOTION_MAP, CHANNEL_NAMES
)


class SEEDLoader:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or DATASET_PATH
        self.raw_eeg_path = RAW_EEG_PATH
        self.n_channels = N_CHANNELS
        self.trial_labels = np.array(TRIAL_LABELS)
        self.subject_sessions = SUBJECT_SESSIONS
        self.channel_names = CHANNEL_NAMES

    def load_channel_positions(self):
        channels = []
        with open(CHANNEL_FILE, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    channels.append({
                        'number': int(parts[0]),
                        'angle': float(parts[1]),
                        'radius': float(parts[2]),
                        'name': parts[3]
                    })
        return channels

    def load_trial_eeg(self, subject_id, session_date, trial_idx):
        """
        Load single trial EEG data.

        Args:
            subject_id: Subject ID (1-15)
            session_date: Session date string (e.g., '20131027')
            trial_idx: Trial index (0-14, 0-based)

        Returns:
            eeg_data: numpy array (62, timepoints) or None
            label: emotion label (-1, 0, 1)
        """
        filename = f"{subject_id}_{session_date}.mat"
        filepath = os.path.join(self.raw_eeg_path, filename)

        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None, None

        data = scipy.io.loadmat(filepath)

        # Find trial key - SEED uses various prefixes
        trial_key = None
        trial_num = trial_idx + 1  # SEED uses 1-based trial numbering

        for key in data.keys():
            if key.startswith('__'):
                continue
            # Match patterns: xxx_eeg1, xxx_eeg_1, eeg1, etc.
            nums = re.findall(r'\d+', key)
            if nums and int(nums[-1]) == trial_num and 'eeg' in key.lower():
                trial_key = key
                break

        if trial_key is None:
            return None, None

        eeg_data = data[trial_key]

        # Ensure shape is (channels, timepoints)
        if eeg_data.shape[0] != self.n_channels:
            if eeg_data.shape[1] == self.n_channels:
                eeg_data = eeg_data.T
            else:
                print(f"Unexpected shape {eeg_data.shape} for {trial_key}")
                return None, None

        label = self.trial_labels[trial_idx]
        return eeg_data, label

    def load_all_trials(self, subject_id, session_date):
        """
        Load all 15 trials for a subject-session.

        Returns:
            trials: list of (eeg_data, label, trial_idx) tuples
        """
        trials = []
        for i in range(15):
            eeg, label = self.load_trial_eeg(subject_id, session_date, i)
            if eeg is not None:
                trials.append((eeg, label, i))
        return trials

    def load_trials_by_emotion(self, subject_id, session_date):
        """
        Load trials grouped by emotion.

        Returns:
            dict: {emotion_name: [(eeg_data, trial_idx), ...]}
        """
        trials = self.load_all_trials(subject_id, session_date)
        by_emotion = {}

        for eeg, label, idx in trials:
            emotion_name = EMOTION_MAP.get(label, 'unknown')
            if emotion_name not in by_emotion:
                by_emotion[emotion_name] = []
            by_emotion[emotion_name].append((eeg, idx))

        return by_emotion

    def get_all_subject_sessions(self):
        """Get list of all (subject_id, session_date) pairs."""
        pairs = []
        for sid, sessions in sorted(self.subject_sessions.items()):
            for sdate in sessions:
                pairs.append((sid, sdate))
        return pairs

    def get_subject_info(self):
        """Load subject ID and gender info."""
        info_file = os.path.join(self.dataset_path, "subject-id-gender-seed.txt")
        subjects = []
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        subjects.append({'id': int(parts[0]), 'gender': parts[1]})
        return subjects


if __name__ == "__main__":
    loader = SEEDLoader()
    print(f"Dataset path: {loader.raw_eeg_path}")
    print(f"Subjects: {len(loader.subject_sessions)}")
    print(f"Trial labels: {loader.trial_labels}")
    print(f"Emotion distribution:")
    for label, name in EMOTION_MAP.items():
        count = np.sum(loader.trial_labels == label)
        print(f"  {name}: {count} trials per session")
    print()

    # Test load one trial
    eeg, label = loader.load_trial_eeg(1, '20131027', 0)
    if eeg is not None:
        print(f"Trial 1: shape={eeg.shape}, label={label} ({EMOTION_MAP[label]})")
        print(f"Duration: {eeg.shape[1]/200:.1f}s @ 200Hz")
