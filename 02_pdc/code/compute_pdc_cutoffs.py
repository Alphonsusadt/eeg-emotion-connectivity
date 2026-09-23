"""
Hitung cutoff PDC global per band (persentil ke-75, sesuai config.PDC_PERCENTILE
dan `phase_1_data_structuring.ipynb::compute_global_pdc_cutoff`), dipool dari
SELURUH subjek/sesi/trial -- bukan per-subjek -- supaya visualisasi topdown
PDC menggunakan definisi "signifikan" yang identik dengan yang menghasilkan
angka density/degree di `cleaned_graph_metrics_pdc_*.csv`.

Hasil disimpan ke `pdc_global_cutoffs.json` di folder ini supaya tidak perlu
dihitung ulang setiap kali render ulang figure.
"""
import json
import os

import numpy as np
from tqdm import tqdm

PDC_DIR = r"D:\Skripsi\new_data\02_pdc\output\pdc_matrices"
PDC_BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
PDC_PERCENTILE = 75
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdc_global_cutoffs.json")


def main():
    subjects = sorted(
        d for d in os.listdir(PDC_DIR)
        if os.path.isdir(os.path.join(PDC_DIR, d)) and d.startswith("subject_")
    )

    cutoffs = {}
    for band in tqdm(PDC_BANDS, desc="Bands"):
        pooled = []
        for subject in subjects:
            sub_path = os.path.join(PDC_DIR, subject)
            sessions = sorted(
                d for d in os.listdir(sub_path)
                if os.path.isdir(os.path.join(sub_path, d)) and d.startswith("session_")
            )
            for session in sessions:
                sess_path = os.path.join(sub_path, session)
                for f in os.listdir(sess_path):
                    if f.startswith(f"pdc_{band}_trial_") and f.endswith(".npy"):
                        m = np.load(os.path.join(sess_path, f))
                        offdiag = m[~np.eye(m.shape[0], dtype=bool)]
                        pooled.append(offdiag)
        pooled = np.concatenate(pooled)
        cutoff = float(np.percentile(pooled, PDC_PERCENTILE))
        cutoffs[band] = cutoff
        print(f"  {band:12s}: cutoff={cutoff:.6f}  (n_pooled={len(pooled)})")

    with open(OUT_PATH, "w") as f:
        json.dump(cutoffs, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
