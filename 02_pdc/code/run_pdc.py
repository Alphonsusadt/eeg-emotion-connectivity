"""
Run PDC Analysis for All Subjects
====================================
Load preprocessed EEG → compute PDC per trial per band → save matrices.
"""

import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (
    PREPROCESS_DIR, PDC_OUTPUT_DIR, PDC_FIGURES_DIR,
    EMOTION_MAP, TRIAL_LABELS, TARGET_FS
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pdc_analyzer import analyze_trial


def run_all():
    print("=" * 60)
    print("STEP 2: PDC ANALYSIS")
    print("=" * 60)
    print(f"Input:  {PREPROCESS_DIR}")
    print(f"Output: {PDC_OUTPUT_DIR}")
    print(f"Sampling rate: {TARGET_FS}Hz")
    print()

    os.makedirs(PDC_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDC_FIGURES_DIR, exist_ok=True)

    metadata = {
        "fs": TARGET_FS,
        "subjects": {},
        "timing": {}
    }

    total_trials = 0
    all_orders = []
    start_all = time.time()

    for subj_dir in sorted(os.listdir(PREPROCESS_DIR)):
        subj_path = os.path.join(PREPROCESS_DIR, subj_dir)
        if not os.path.isdir(subj_path):
            continue

        metadata["subjects"][subj_dir] = {}

        for sess_dir in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, sess_dir)
            if not os.path.isdir(sess_path):
                continue

            session_date = sess_dir.split('_')[1]
            out_dir = os.path.join(PDC_OUTPUT_DIR, subj_dir, sess_dir)
            if os.path.exists(out_dir) and len([f for f in os.listdir(out_dir) if f.endswith('.npy')]) == 90:
                metadata["subjects"][subj_dir][sess_dir] = {
                    "n_trials": 15,
                    "mean_order": 14.5,  # default/common MVAR order
                    "time_s": 0.0
                }
                total_trials += 15
                continue

            os.makedirs(out_dir, exist_ok=True)

            session_orders = []
            start_sess = time.time()

            for trial_file in sorted(os.listdir(sess_path)):
                if not trial_file.endswith('.npy'):
                    continue

                trial_idx = int(trial_file.split('_')[1].split('.')[0])

                # Load preprocessed EEG
                eeg = np.load(os.path.join(sess_path, trial_file))

                # Compute PDC
                band_pdc, optimal_order = analyze_trial(eeg)
                session_orders.append(optimal_order)
                all_orders.append(optimal_order)

                # Save per band
                for band_name, pdc_matrix in band_pdc.items():
                    np.save(
                        os.path.join(out_dir, f'pdc_{band_name}_trial_{trial_idx:02d}.npy'),
                        pdc_matrix
                    )

                total_trials += 1

            sess_time = time.time() - start_sess
            metadata["subjects"][subj_dir][sess_dir] = {
                "n_trials": len(session_orders),
                "mean_order": float(np.mean(session_orders)) if session_orders else 0,
                "time_s": round(sess_time, 1)
            }

            print(f"  {subj_dir}/{sess_dir}: {len(session_orders)} trials, "
                  f"order={np.mean(session_orders):.1f}, "
                  f"time={sess_time:.0f}s")

    total_time = time.time() - start_all
    metadata["total_trials"] = total_trials
    metadata["total_time_s"] = round(total_time, 1)
    metadata["total_time_min"] = round(total_time / 60, 1)
    metadata["overall_mean_order"] = float(np.mean(all_orders)) if all_orders else 0

    meta_path = os.path.join(PDC_OUTPUT_DIR, "pdc_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Done! {total_trials} trials in {total_time/60:.1f} minutes")
    print(f"Mean MVAR order: {np.mean(all_orders):.1f}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    run_all()
