# Pipeline Lengkap: GC + PDC untuk Klasifikasi Emosi EEG (SEED Dataset)

> **Skripsi:** Perbandingan Granger Causality (GC) dan Partial Directed Coherence (PDC)
> pada analisis konektivitas otak untuk pengenalan emosi menggunakan dataset SEED.

---

## 🎯 Konteks & Tujuan Penelitian (Ringkasan Bab 1–3)

> **Catatan ini ditujukan untuk siapapun (termasuk AI agent) yang akan menulis kode
> dalam pipeline ini, agar setiap keputusan teknis selalu selaras dengan tujuan penelitian.**

### Latar Belakang & Masalah yang Ingin Dipecahkan

Penelitian ini lahir dari sebuah keterbatasan mendasar pada metode deteksi emosi EEG yang
ada saat ini. Mayoritas penelitian menggunakan **fitur spektral lokal** seperti *Power Spectral
Density (PSD)* dan *Differential Entropy (DE)* yang hanya mengukur "seberapa aktif setiap
region otak", **tanpa mempertanyakan "bagaimana region-region tersebut saling berkomunikasi"**.

Padahal, neurosains telah membuktikan bahwa emosi diproses melalui dua mekanisme jaringan
yang bersifat **terarah (directed)**:
- **Bottom-up processing** — sinyal emosional dari amygdala/limbik naik ke korteks.
- **Top-down regulation** — *prefrontal cortex* mengontrol dan meregulasi respons limbik.

Keterbatasan ini menyebabkan model yang dibangun **sulit diinterpretasikan secara
neurobiologis** — tidak jelas wilayah otak mana yang paling berperan atau bagaimana pola
interaksi antar-region membentuk pengalaman emosional yang berbeda.

### Solusi yang Diusulkan

Penelitian ini mengintegrasikan analisis **konektivitas efektif terarah (directed effective
connectivity)** menggunakan dua metode komplementer:

| Metode | Domain | Cara Kerja |
|---|---|---|
| **Granger Causality (GC)** | Waktu | Mengukur apakah riwayat sinyal kanal A membantu memprediksi kanal B (bivariat, FDR-corrected) |
| **Partial Directed Coherence (PDC)** | Frekuensi | Mengukur aliran informasi terarah per pita frekuensi menggunakan model MVAR |

Dari matriks konektivitas 62×62 yang dihasilkan, diekstrak **metrik topologi graf terarah**
sebagai fitur: *in/out-degree*, *in/out-strength*, *density*, *hub channel*, dsb.

### Tiga Tujuan Utama Penelitian

1. **Memetakan pola konektivitas** — Identifikasi bagaimana wilayah Prefrontal, Temporal,
   dan Central berkomunikasi secara terarah saat memproses emosi *negative/neutral/positive*.

2. **Membandingkan GC vs PDC** — Evaluasi metode mana yang lebih optimal sebagai fitur
   klasifikasi emosi (hipotesis: fitur konektivitas > fitur spektral tradisional).

3. **Validasi lintas subjek** — Uji kemampuan generalisasi model menggunakan
   *Leave-One-Subject-Out Cross-Validation (LOSO-CV)*.

### Target Performa yang Ingin Dicapai

| Skenario | Target Akurasi |
|---|---|
| Fitur konektivitas saja (GC atau PDC) | **70–80%** |
| Fitur gabungan (GC + PDC + spektral) | **75–85%** |
| Validasi lintas dataset (SEED→DEAP) | **≥ 65%** |

### Detail Dataset & Setup Eksperimen

- **Dataset:** SEED (Shanghai Jiao Tong University) — dataset EEG emosi *open-access*.
- **Subjek:** 15 subjek (7 pria, 8 wanita), **3 sesi** perekaman di hari berbeda.
- **Stimulus:** 15 klip video emosional per sesi → label: `negative`, `neutral`, `positive`.
- **EEG:** 62 kanal, di-*downsample* ke **100 Hz**, *bandpass filter* 1–45 Hz.
- **Total trial:** 15 subjek × 3 sesi × 15 trial = **675 trial independen**.
- **Klasifier yang diuji:** LDA, SVM, Random Forest, XGBoost, Ensemble, GCN/GAT.
- **Evaluasi:** Akurasi, Precision, Recall, F1-score per kelas, uji statistik (paired t-test /
  Wilcoxon), McNemar test, dan *Intraclass Correlation Coefficient (ICC)*.

### Konvensi Kolom CSV yang Wajib Dipahami

Setiap CSV output dari **Phase 1** memiliki dua kolom label emosi dengan makna berbeda:

| Kolom | Isi | Kegunaan |
|---|---|---|
| `class_label` | Teks: `'negative'`, `'neutral'`, `'positive'` | **Grouping & plotting** di Phase 2–3 |
| `class` | Numerik: `0`, `1`, `2` | **Input classifier** di Phase 4–6 |

> ⚠️ **Penting untuk agent yang menulis kode:** Selalu gunakan `class_label` untuk
> operasi `groupby`, `reindex`, dan visualisasi. Gunakan `class` (numerik) hanya sebagai
> target label untuk training model ML. **Jangan pernah mengubah file CSV secara langsung —
> selalu perbaiki kode yang membacanya.**

---


## Alur Pipeline (Hybrid: GC Concatenated + PDC Per-Trial ROI)

```
DATASET (D:\Skripsi\dataset\SEED)
    │
    ▼
┌──────────────────────────────┐
│  00_preprocessing            │  Raw EEG → filter → downsample → per-trial .npy
│  Bandpass 1-45Hz, Notch 50Hz │  Full 62 channels
│  Downsample 200→100Hz        │
└──────────┬───────────────────┘
           │
     ┌─────┴──────────┐
     ▼                ▼
┌──────────────┐  ┌──────────────────┐
│ 01_GC        │  │ 02_PDC           │
│ 62 channels  │  │ 62 channels      │
│ PER-TRIAL    │  │ PER-TRIAL        │
│ FDR threshold│  │ MVAR AIC 3-15    │
│ ~630 samples │  │ 5 bands          │
│              │  │ ~630 samples     │
└──────┬───────┘  └────────┬─────────┘
       │                   │
       └───────┬───────────┘
               ▼
┌──────────────────────────────┐
│  phase_1_data_structuring    │  Aggregate → df_gc (~630), df_pdc (~630), df_combined
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  phase_2_statistics          │  Descriptive stats
│  Per-person, per-class,      │  Uji normalitas (Shapiro-Wilk) & homogenitas (Levene)
│  inferential, GC vs PDC      │  ANOVA/Kruskal-Wallis, Post-hoc, effect size
│  Boxplot, violin plot        │  Perbandingan GC vs PDC
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  phase_3_feature_engineering │  Asymmetry index, ratio, normalized
│  Derived features, selection │  SelectKBest, MI, PCA
│  MI & F-score comparison     │  GC vs PDC informativeness
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  phase_4_classification      │  LDA, SVM, RF, XGBoost, Ensemble
│  Multi-model comparison      │  5 models × 3 inputs (GC/PDC/Combined)
│  Hakim: GC vs PDC            │  GridSearchCV, McNemar test
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  phase_5_loso                │  Leave-One-Subject-Out CV
│  Cross-subject               │  LOSO semua model × semua input
│  ICC analysis                │  Subject-wise normalization
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  phase_6_deep_learning       │  GCN, GAT, CNN, LSTM, Transformer
│  Graph neural networks       │  Late Fusion (dual-branch GCN)
│  Separate + Fusion           │  LOSO untuk DL
└──────────────────────────────┘
```

---

## Struktur Folder

```
D:\Skripsi\new_data\
│
├── config.py                              # Konfigurasi global (path, parameter)
├── readme.md                              # Dokumentasi (file ini)
│
│
├── 00_preprocessing\                      # Load EEG → filter → downsample → save per trial
│   ├── code\
│   │   ├── seed_loader.py                 #   Load SEED dataset (subject, session, trial, label)
│   │   ├── preprocess.py                  #   Bandpass + notch + downsample
│   │   └── run_preprocess.py              #   Script utama: proses semua subjek
│   └── output\
│       └── preprocessed_eeg\              #   Output: .npy per trial
│           ├── subject_01\
│           │   ├── session_20131027\
│           │   │   ├── trial_01.npy       #     shape: (62, timepoints), fs=100Hz
│           │   │   ├── trial_02.npy
│           │   │   └── ... (15 trials)
│           │   ├── session_20131030\
│           │   └── session_20131107\
│           ├── subject_02\
│           └── ... (15 subjects)
│
│
├── 01_granger_causality\                  # Hitung GC per trial
│   ├── code\
│   │   ├── gc_analyzer.py                 #   GC pairwise + FDR threshold
│   │   └── run_gc.py                      #   Script utama: proses semua subjek
│   └── output\
│       ├── gc_matrices\                   #   Output: GC matrices per trial
│       │   ├── subject_01\
│       │   │   └── session_20131027\
│       │   │       ├── gc_raw_trial_01.npy       # (62,62) F-statistic values
│       │   │       ├── p_values_trial_01.npy     # (62,62) p-values
│       │   │       ├── gc_thresholded_trial_01.npy  # (62,62) setelah FDR
│       │   │       ├── gc_raw_trial_02.npy
│       │   │       └── ...
│       │   └── ... (15 subjects)
│       ├── metadata.json                  #   Config & timing summary
│       └── figures\                       #   Heatmap GC per emosi
│
│
├── 02_pdc\                                # Hitung PDC per trial
│   ├── code\
│   │   ├── pdc_analyzer.py                #   MVAR fit + PDC computation, 5 bands
│   │   └── run_pdc.py                     #   Script utama: proses semua subjek
│   └── output\
│       ├── pdc_matrices\                  #   Output: PDC matrices per trial per band
│       │   ├── subject_01\
│       │   │   └── session_20131027\
│       │   │       ├── pdc_delta_trial_01.npy    # (62,62) per band
│       │   │       ├── pdc_theta_trial_01.npy
│       │   │       ├── pdc_alpha_trial_01.npy
│       │   │       ├── pdc_beta_trial_01.npy
│       │   │       ├── pdc_gamma_trial_01.npy
│       │   │       ├── pdc_broadband_trial_01.npy
│       │   │       └── ...
│       │   └── ... (15 subjects)
│       ├── metadata.json
│       └── figures\                       #   Heatmap PDC per band per emosi
│
│
├── phase_1_data_structuring\              # Phase 1: Data Structuring & Quality Check
│   ├── code\
│   │   └── phase1_data_structuring.ipynb  #   Aggregate + clean
│   ├── csv\                               #   Output: 8 CSV bersih
│   │   ├── cleaned_graph_metrics_gc.csv           # subject, class, in/out degree, strength...
│   │   ├── cleaned_graph_metrics_pdc_alpha.csv
│   │   ├── cleaned_graph_metrics_pdc_beta.csv
│   │   ├── cleaned_graph_metrics_pdc_gamma.csv
│   │   ├── cleaned_graph_metrics_pdc_delta.csv
│   │   ├── cleaned_graph_metrics_pdc_theta.csv
│   │   ├── cleaned_graph_metrics_pdc_broadband.csv
│   │   └── cleaned_graph_metrics_combined.csv     # GC + PDC gabungan
│   └── figures\
│
│
├── phase_2_statistics\                    # Phase 2: Analisis Statistik
│   ├── code\
│   │   └── phase2_statistics.ipynb
│   ├── 2.1_individual_analysis\           #   Per-person mean/std, line plot
│   ├── 2.2_class_analysis\               #   Violin/box plot per fitur per kelas
│   ├── 2.3_inferential_statistics\       #   ANOVA/Kruskal, post-hoc, effect size
│   ├── 2.4_gc_vs_pdc_comparison\         #   Side-by-side GC vs PDC
│   └── figures\
│
│
├── phase_3_feature_engineering\           # Phase 3: Feature Engineering
│   ├── code\
│   │   └── phase3_feature_engineering.ipynb
│   ├── 3.1_derived_features\             #   Asymmetry index, ratio, normalized, log
│   ├── 3.2_feature_selection\            #   SelectKBest, MI, PCA
│   ├── 3.3_informativeness_comparison\   #   MI & F-score GC vs PDC
│   ├── csv\                               #   Output: engineered features (8 CSV)
│   │   ├── engineered_features_gc.csv
│   │   ├── engineered_features_pdc_delta.csv
│   │   ├── engineered_features_pdc_theta.csv
│   │   ├── engineered_features_pdc_alpha.csv
│   │   ├── engineered_features_pdc_beta.csv
│   │   ├── engineered_features_pdc_gamma.csv
│   │   ├── engineered_features_pdc_broadband.csv
│   │   └── engineered_features_combined.csv
│   └── figures\
│
│
├── phase_4_classification\                # Phase 4: Multi-Model Classification
│   ├── code\
│   │   └── phase4_classification.ipynb
│   ├── 4.1_lda\                          #   LDA scatter plot, coefficients
│   ├── 4.2_svm\                          #   SVM RBF/Linear/Poly
│   ├── 4.3_random_forest\                #   RF + feature importance
│   ├── 4.4_xgboost\                      #   XGBoost
│   ├── 4.5_ensemble\                     #   Stacking + Voting
│   ├── 4.6_hyperparameter_tuning\        #   Optuna/GridSearchCV
│   ├── 4.7_gc_vs_pdc_comparison\         #   Tabel perbandingan + significance test
│   └── figures\
│
│
├── phase_5_loso\                          # Phase 5: Cross-Subject LOSO
│   ├── code\
│   │   └── phase5_loso.ipynb
│   ├── 5.1_loso_cv\                      #   LOSO implementasi
│   ├── 5.2_loso_all_models\              #   LOSO semua model × 3 input
│   ├── 5.3_loso_analysis\                #   Bar plot per subjek, pooled vs LOSO
│   ├── 5.4_gc_vs_pdc_loso\              #   LOSO GC vs PDC significance
│   ├── 5.5_icc_analysis\                #   Intraclass Correlation Coefficient
│   ├── 5.6_subject_normalization\        #   Subject-wise normalization
│   └── figures\
│
│
└── phase_6_deep_learning\                 # Phase 6: Deep Learning
    ├── code\
    │   └── phase6_deep_learning.ipynb
    ├── 6.1_separate_branch\              #   GCN, GAT, CNN, LSTM (GC vs PDC terpisah)
    ├── 6.2_late_fusion\                  #   Dual-branch GCN (GC + PDC)
    ├── 6.3_multi_channel\               #   CNN 2-channel input
    ├── 6.4_transformer\                  #   Self-attention
    ├── 6.5_evaluation\                   #   Tabel perbandingan semua arsitektur
    ├── 6.6_loso_dl\                      #   LOSO untuk model DL terbaik
    ├── models\                           #   Saved model weights
    └── figures\
```

---

## Perubahan dari Pipeline Lama

| Aspek | Lama | Baru (Hybrid) |
|---|---|---|
| Downsampling | 50Hz (factor 4) | **100Hz (factor 2)** |
| Preprocessing | Tidak ada filtering | **Bandpass 1-45Hz + notch 50Hz** |
| GC Threshold | p < 0.05 (density 98%) | **FDR correction (target density <30%)** |
| GC Channels | 62 | **62 (full, per-trial)** |
| GC Sample Size | 135 | **~630 (per-trial, 15 subj × 3 sesi × ~14 trial)** |
| PDC Channels | 62 | **62 (full channels)** |
| PDC Granularity | Concatenated (135) | **Per-trial (~630)** |
| PDC MVAR Order | 3 (fixed) | **3-15 (AIC optimal)** |
| PDC Bands | 3 (alpha, beta, broadband) | **6 (delta, theta, alpha, beta, gamma, broadband)** |
| Feature Extraction | Graph metrics (320 fitur) | **Upper-triangle + graph metrics** |
| Model | SVM saja | **LDA, SVM, RF, XGBoost, Ensemble, DL** |
| Validasi | LOSO saja | **LOSO + ICC + subject normalization** |
| Deep Learning | Tidak ada | **CNN, Dual-Branch Late Fusion** |

## Kenapa Hybrid?

| Masalah | Solusi |
|---|---|
| GC 62ch per-trial = komputasi lama | GC **per-trial** FDR threshold, ~630 samples |
| MVAR 62ch per-trial | PDC **62 channels** per-trial, AIC order selection |
| Deep Learning ~630 samples | CNN/GCN pada matrix, ekspektasi realistis |

---

## Urutan Eksekusi

```
1.  00_preprocessing/code/run_preprocess.py       # ~2 jam
2.  01_granger_causality/code/run_gc.py            # ~20 jam (15 subj × 3 sesi × 15 trial)
3.  02_pdc/code/run_pdc.py                        # ~2 jam
4.  phase_1_data_structuring/code/*.ipynb          # ~30 menit
5.  phase_2_statistics/code/*.ipynb                # ~1 jam
6.  phase_3_feature_engineering/code/*.ipynb       # ~1 jam
7.  phase_4_classification/code/*.ipynb            # ~3 jam
8.  phase_5_loso/code/*.ipynb                      # ~3 jam
9.  phase_6_deep_learning/code/*.ipynb             # ~1 hari
```

---

## Format Data Antar Step

### 00 → 01/02 (preprocessed EEG per trial)
```
File: trial_XX.npy
Shape: (62, timepoints)    # full 62 channels, ~23000 samples @ 100Hz
Dtype: float64
```

### 01 → phase_1 (GC matrices, PER-TRIAL per subject-session-trial)
```
File: gc_thresholded_trial_{XX}.npy  # (62, 62) F-statistic after FDR, per trial
File: gc_raw_trial_{XX}.npy          # (62, 62) raw F-statistic
File: p_values_trial_{XX}.npy        # (62, 62) p-values
Total: 15 subjects x 3 sessions x 15 trials = ~630 matrices
```

### 02 → phase_1 (PDC matrices, PER-TRIAL with 62 channels)
```
File: pdc_{band}_trial_{XX}.npy   # (62, 62) per band, per trial
Total: 15 subjects x 3 sessions x 15 trials = ~630 matrices per band
Bands: delta, theta, alpha, beta, gamma, broadband
```

### phase_1 → phase 2-6 (tabular CSV)
```
GC (~630 rows):
subject_id | session | trial | class | class_label |
mean_out_degree | mean_in_degree | ... | gc_density

PDC (~630 rows):
subject_id | session | trial | class | class_label |
mean_out_degree | mean_in_degree | ... | pdc_density
```
