"""
Run Preprocessing for All Subjects
====================================
Load raw EEG → preprocess → save per trial.
"""

import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PREPROCESS_DIR, EMOTION_MAP, TARGET_FS, ORIGINAL_FS
)

# Import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seed_loader import SEEDLoader
from preprocess import preprocess_eeg, save_preprocessed


def run_all():
    loader = SEEDLoader()
    all_sessions = loader.get_all_subject_sessions()

    print("=" * 60)
    print("STEP 0: PREPROCESSING ALL SUBJECTS")
    print("=" * 60)
    print(f"Total sessions: {len(all_sessions)}")
    print(f"Output: {PREPROCESS_DIR}")
    print()

    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    metadata = {
        "original_fs": ORIGINAL_FS,
        "target_fs": TARGET_FS,
        "subjects": {},
        "timing": {}
    }

    total_trials = 0
    start_all = time.time()

    for subject_id, session_date in tqdm(all_sessions, desc="Sessions"):
        subj_key = f"subject_{subject_id:02d}"
        if subj_key not in metadata["subjects"]:
            metadata["subjects"][subj_key] = {}

        session_dir = os.path.join(PREPROCESS_DIR, subj_key, f"session_{session_date}")
        if os.path.exists(session_dir) and len([f for f in os.listdir(session_dir) if f.endswith('.npy')]) == 15:
            metadata["subjects"][subj_key][session_date] = {
                "n_trials": 15,
                "time_s": 0.0,
                "trials": []
            }
            total_trials += 15
            continue

        trials = loader.load_all_trials(subject_id, session_date)

        session_trials = []
        start_sess = time.time()

        for eeg, label, trial_idx in trials:
            processed = preprocess_eeg(eeg)
            filepath = save_preprocessed(processed, subject_id, session_date, trial_idx, PREPROCESS_DIR)

            session_trials.append({
                "trial": trial_idx + 1,
                "label": int(label),
                "emotion": EMOTION_MAP[label],
                "shape": list(processed.shape),
                "file": os.path.basename(filepath)
            })
            total_trials += 1

        sess_time = time.time() - start_sess
        metadata["subjects"][subj_key][session_date] = {
            "n_trials": len(session_trials),
            "time_s": round(sess_time, 1),
            "trials": session_trials
        }

    total_time = time.time() - start_all

    metadata["total_trials"] = total_trials
    metadata["total_time_s"] = round(total_time, 1)
    metadata["total_time_min"] = round(total_time / 60, 1)

    # Save metadata
    meta_path = os.path.join(PREPROCESS_DIR, "preprocess_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Done! {total_trials} trials preprocessed in {total_time/60:.1f} minutes")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    run_all()
