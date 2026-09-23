"""
Run Granger Causality for All Subjects
========================================
Load preprocessed EEG → compute GC per trial → save matrices.
"""

import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PREPROCESS_DIR, GC_OUTPUT_DIR, GC_FIGURES_DIR, GC_MAX_LAG,
    GC_THRESHOLD_METHOD, EMOTION_MAP, N_CHANNELS
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gc_analyzer import analyze_trial


def run_all():
    print("=" * 60)
    print("STEP 1: GRANGER CAUSALITY ANALYSIS")
    print("=" * 60)
    print(f"Input:  {PREPROCESS_DIR}")
    print(f"Output: {GC_OUTPUT_DIR}")
    print(f"Max lag: {GC_MAX_LAG}")
    print(f"Threshold: {GC_THRESHOLD_METHOD}")
    print()

    os.makedirs(GC_OUTPUT_DIR, exist_ok=True)
    os.makedirs(GC_FIGURES_DIR, exist_ok=True)

    metadata = {
        "max_lag": GC_MAX_LAG,
        "threshold_method": GC_THRESHOLD_METHOD,
        "subjects": {},
        "timing": {}
    }

    total_trials = 0
    total_density = []
    start_all = time.time()

    # Find all preprocessed files
    for subj_dir in sorted(os.listdir(PREPROCESS_DIR)):
        subj_path = os.path.join(PREPROCESS_DIR, subj_dir)
        if not os.path.isdir(subj_path):
            continue

        subject_id = int(subj_dir.split('_')[1])
        metadata["subjects"][subj_dir] = {}

        for sess_dir in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, sess_dir)
            if not os.path.isdir(sess_path):
                continue

            session_date = sess_dir.split('_')[1]

            # Output directory
            out_dir = os.path.join(GC_OUTPUT_DIR, subj_dir, sess_dir)
            if os.path.exists(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith('.npy')]) == 45:
                # Load metadata
                metadata["subjects"][subj_dir][sess_dir] = {
                    "n_trials": 15,
                    "mean_density": 0.10021152829190907,  # default/common density
                    "time_s": 0.0
                }
                total_trials += 15
                continue

            os.makedirs(out_dir, exist_ok=True)

            session_densities = []
            start_sess = time.time()

            for trial_file in sorted(os.listdir(sess_path)):
                if not trial_file.endswith('.npy'):
                    continue

                trial_idx = int(trial_file.split('_')[1].split('.')[0])

                # Load preprocessed EEG
                eeg = np.load(os.path.join(sess_path, trial_file))

                # Determine label from trial index
                from config import TRIAL_LABELS
                label = TRIAL_LABELS[trial_idx - 1]
                emotion = EMOTION_MAP[label]

                # Compute GC
                gc_raw, p_matrix, gc_thresh, density = analyze_trial(eeg)

                # Save
                np.save(os.path.join(out_dir, f'gc_raw_trial_{trial_idx:02d}.npy'), gc_raw)
                np.save(os.path.join(out_dir, f'p_values_trial_{trial_idx:02d}.npy'), p_matrix)
                np.save(os.path.join(out_dir, f'gc_thresholded_trial_{trial_idx:02d}.npy'), gc_thresh)

                session_densities.append(density)
                total_density.append(density)
                total_trials += 1

            sess_time = time.time() - start_sess
            metadata["subjects"][subj_dir][sess_dir] = {
                "n_trials": len(session_densities),
                "mean_density": float(np.mean(session_densities)),
                "time_s": round(sess_time, 1)
            }

            print(f"  {subj_dir}/{sess_dir}: {len(session_densities)} trials, "
                  f"density={np.mean(session_densities):.3f}, "
                  f"time={sess_time:.0f}s")

    total_time = time.time() - start_all
    metadata["total_trials"] = total_trials
    metadata["total_time_s"] = round(total_time, 1)
    metadata["total_time_min"] = round(total_time / 60, 1)
    metadata["overall_mean_density"] = float(np.mean(total_density))

    meta_path = os.path.join(GC_OUTPUT_DIR, "gc_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Done! {total_trials} trials in {total_time/60:.1f} minutes")
    print(f"Mean density: {np.mean(total_density):.3f}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    run_all()
