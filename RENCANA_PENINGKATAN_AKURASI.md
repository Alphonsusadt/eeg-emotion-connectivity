# Rencana: Menaikkan Akurasi Klasifikasi Emosi ke Target 80%+ (fokus `new_data/`, s.d. Phase 4)

## 📌 STATUS TERKINI (update terakhir: 2026-07-21, sesi malam)

**Progres: Langkah 1 ✅ — Langkah 2 ✅ — Langkah 3 ✅ — Langkah 4 ✅ (ROI aggregation, selesai — hasil terbaik baru: 75.56%)**

### Langkah 4 — ROI Aggregation (sesi ini, lanjutan)
Opsi yang dipilih user dari 4 opsi Langkah 4: **ROI aggregation** (bukan class-weighting,
model tambahan, atau menerima plateau 75%).

1. **Phase 3.4 (`phase_3_feature_engineering/3.4_roi_aggregation/`) — sudah dibuat & dijalankan
   sebelumnya** (ditemukan sudah ada di awal sesi ini, belum tercatat di dokumen ini). Mengagregasi
   matriks GC/PDC 62x62 per-channel jadi 6x6 antar-region (Prefrontal, Frontal, Temporal, Central,
   Parietal, Occipital) berdasar `CHANNEL_NAMES` di `config.py`. Output: `engineered_features_gc_roi.csv`,
   `engineered_features_pdc_roi_<band>.csv`, `engineered_features_roi_gc_pdc.csv` (GC+PDC ROI tanpa
   spectral), `engineered_features_roi_all.csv` (GC+PDC ROI+Spectral).
   - Quick baseline (RF tunggal, grouped CV): GC+PDC_ROI+Spectral **71.85% ± 4.16%** (990 fitur) —
     lihat `output/roi_quick_baseline.csv`.
   - **Temuan kunci (apples-to-apples RF vs RF):** ROI (990 fitur) **mengalahkan** full per-channel
     Combined RF (70.81%, 2676 fitur, dari `4.3_random_forest/output/rf_results.csv`) meski dimensi
     fiturnya sepertiganya. Sinyal kuat bahwa agregasi ROI layak diteruskan ke pipeline Phase 4 penuh.
2. **Wiring ke 4.1-4.7** (dikerjakan sesi ini): 4 dataset ROI baru (`GC_ROI`, `PDC_ROI`,
   `GC+PDC_ROI`, `GC+PDC_ROI+Spectral`, load langsung dari CSV Phase 3.4 di atas) ditambahkan ke
   `datasets` dict di `4.1_lda`, `4.2_svm`, `4.3_random_forest`, `4.4_xgboost`, `4.5_ensemble`,
   `4.6_hyperparameter_tuning` (cell load + cell datasets, identik di 6 notebook). `4.7` tidak perlu
   diubah (auto-agregasi via glob nama dataset dari cv_scores.csv tiap sub-phase).
   - Hasil per model tunggal (grouped CV) setelah rerun:

     | Model | Dataset terbaik (ROI) | Akurasi ROI | Dataset terbaik (non-ROI, utk banding) | Akurasi non-ROI |
     |---|---|---|---|---|
     | LDA | GC+PDC_ROI+Spectral | 66.8% | Spectral | 69.8% |
     | SVM_RBF | GC+PDC_ROI+Spectral | 73.5% | Combined / PDC_Theta+Spectral | 73.6% (seri) |
     | RF | **GC+PDC_ROI+Spectral** | **71.7%** | Combined | 70.8% (ROI menang) |
     | XGBoost, Ensemble/Stacking, Tuned | *(sedang jalan / belum)* | — | — | — |

   - **Catatan progres:** 4.1, 4.2, 4.3 selesai. 4.4 (XGBoost) & 4.5 (Ensemble) dijalankan
     background paralel. 4.6 (GridSearchCV tuning) belum dijalankan — grid pakai
     `SelectKBest(k=min(1200, n_features))` jadi ROI (56-990 fitur, semua <1200) seharusnya LEBIH
     CEPAT per fit daripada Combined (2676→dipotong 1200), meski jumlah dataset naik dari 7 ke 11.
     Rencana: jalankan 4.6 setelah 4.4/4.5 selesai, lalu 4.7 untuk agregasi + uji signifikansi final.
   - **SELESAI.** 4.4, 4.5, 4.6 sempat 2x mati (background job killed) saat dijalankan paralel
     (2 job `n_jobs=-1` bersamaan tampaknya memicu resource kill) — diperbaiki dengan menambah
     checkpointing per (dataset, model) di 4.5 & 4.6 (skip pasangan yang sudah tersimpan di CSV
     output, resume otomatis) dan menjalankan job berat satu per satu (sekuensial), bukan paralel.

   **Hasil akhir (grouped CV per subjek, jujur) — `4.7/output/master_comparison_table.csv`, setelah ROI diikutkan:**

   | Rank | Model | Dataset | Akurasi | F1 | Dimensi |
   |---|---|---|---|---|---|
   | 1 | **XGBoost_Tuned** | **GC+PDC_ROI+Spectral** | **75.56% ± 4.68%** | 75.38% | 990 |
   | 2 | Stacking | Combined (per-channel) | 75.41% ± 3.68% | 75.39% | 2676 |
   | 3 | Voting_Soft / SVM_Tuned | GC+PDC_ROI+Spectral | 74.96% ± 6.0%/4.7% | 74.9-75.0% | 990 |
   | 4 | Stacking | GC+PDC_Theta+Spectral | 74.81% | 74.74% | — |

   **Temuan kunci (uji signifikansi paired t-test, `significance_tests.csv`):**
   - Untuk **setiap** model (SVM_Tuned, XGBoost_Tuned, RF, Stacking, Voting, dst.), selisih akurasi
     `Combined` (per-channel, 2676 fitur) vs `GC+PDC_ROI+Spectral` (region-level, 990 fitur) **TIDAK
     signifikan** (p selalu > 0.37, kebanyakan > 0.5) — dan untuk model terbaik (XGBoost_Tuned,
     SVM_Tuned, RF, Voting_Soft) rata-rata ROI justru sedikit **lebih tinggi** dari Combined.
   - Artinya: **agregasi ROI mencapai performa yang setara secara statistik dengan fitur
     per-channel penuh, dengan ~63% lebih sedikit dimensi fitur** (990 vs 2676) — kesimpulan yang
     kuat untuk skripsi: bukan cuma "tidak kalah", tapi lebih parsimonious + langsung menjawab
     tujuan riset #1 (komunikasi antar-ROI: Prefrontal/Temporal/Central).
   - ROI tanpa Spectral (GC_ROI, PDC_ROI, GC+PDC_ROI saja) masih jauh di bawah (58-65%) —
     mengonfirmasi ulang temuan Langkah 3 bahwa Spectral adalah kontributor utama, ROI vs
     per-channel cuma soal bagaimana konektivitas direpresentasikan.
   - Plateau keseluruhan tetap di **~75-76%**, konsisten dengan Langkah 3 — target 80% kemungkinan
     besar tidak tercapai dengan feature engineering lebih lanjut di ruang lingkup ini (lihat
     diskusi dengan user 2026-07-21/22: target dilonggarkan, fokus ke "semaksimal mungkin" bukan
     angka 80% mutlak).

   **Rekomendasi untuk Bab 3/4 skripsi:** laporkan `GC+PDC_ROI+Spectral + XGBoost_Tuned (75.56%)`
   sebagai hasil utama (bukan Combined per-channel) — akurasi setara/lebih baik, dimensi fitur jauh
   lebih kecil (990 vs 2676, less overfitting risk, p≫n lebih terkendali), dan representasinya
   (region-to-region) langsung relevan untuk interpretasi neurobiologis di tujuan riset #1.

### Yang sudah dikerjakan sesi ini
1. **Langkah 1 (Fitur Spektral)** — dibuat & dijalankan penuh. Notebook baru `03_spectral_features/code/03_spectral_features.ipynb` menghasilkan `03_spectral_features/csv/cleaned_spectral_features.csv` (675×626: PSD band power + Differential Entropy, 62 channel × 5 band). Sanity check: fitur paling diskriminatif ada di channel temporal (T7/T8/FT7/FT8/TP7) band beta/gamma — masuk akal secara neurofisiologis.
2. **Langkah 2 (Reduksi Dimensi Fitur)** — notebook `phase_3_feature_engineering/3.1_derived_features/code/phase_3.1_derived_features.ipynb` diubah untuk membuang kolom raw edge-level (`_edge_...`) dari output default dan menambahkan Spectral sebagai metode ke-8. Hasil: `engineered_features_gc.csv` 4078→**296 kolom**, `engineered_features_combined.csv` 28510→**2676 kolom** (versi lengkap dgn edges tetap disimpan terpisah di `engineered_features_full_*.csv` untuk eksperimen opsional).
   - `phase_3_feature_engineering/3.3_informativeness_comparison/code/phase_3.3_informativeness_comparison.ipynb` di-update untuk load dari fitur reduced ini (bukan lagi dari Phase 1 mentah) dan sudah dijalankan ulang. Hasil baseline RF **jujur** (`StratifiedGroupKFold` per subjek), per jenis fitur SENDIRI-SENDIRI (belum digabung):

     | Method | #Fitur | CV Acc (grouped) |
     |---|---|---|
     | GC | 261 | 53.8% ± 7.8% |
     | Spectral (baru) | 382 | 53.3% ± 2.5% |
     | PDC (6 band) | 174-185 | 45-49% |

   - **Temuan kunci:** akurasi 76% yang pernah tercatat di Phase 4 (`master_comparison_table.csv`) itu **inflated akibat data leakage** (CV tidak di-group per subjek). Baseline jujur GC-only sekitar 54%, bukan 76%. Ini bukan kemunduran — ini angka yang benar untuk mulai mengejar target 80%.

3. **Langkah 3 (Perbaikan Evaluasi Phase 4) — SELESAI.** Ditemukan bahwa `phase_4_classification/code/phase_4_classification.ipynb` (root) sudah **legacy/tidak dipakai** — pipeline nyata yang di-agregasi oleh `4.7` adalah 6 sub-notebook modular `4.1_lda` s.d. `4.6_hyperparameter_tuning`, dan keenamnya **sudah lebih dulu memakai** `StratifiedGroupKFold(groups=subject_id)` + subject-wise z-score normalization (baked-in di `prepare_features()`, bukan varian opsional). Yang benar-benar dikerjakan sesi ini:
   - Menambahkan 4 kombinasi fitur baru ke `datasets` di keenam sub-notebook: `Spectral`, `GC+Spectral`, `PDC_<band>+Spectral`, `GC+PDC_<band>+Spectral` (merge on key columns, fitur di-prefix `gc_`/`pdc_`/`spec_` untuk hindari collision nama kolom).
   - Rerun 4.1 → 4.7 penuh dengan CSV fitur baru (hasil reduksi Langkah 2). 4.6 (GridSearchCV, 7 dataset × 3 model) sempat timeout 2x di background (1 jam lalu 2 jam) — akhirnya diselesaikan oleh user secara manual.

   **Hasil akhir (grouped CV per subjek, jujur) — `4.7/output/master_comparison_table.csv`:**

   | Rank | Model | Dataset | Akurasi | F1 |
   |---|---|---|---|---|
   | 1 | Stacking | Combined (GC+6 band PDC+Spectral) | **75.4% ± 3.7%** | 75.4% |
   | 2 | Stacking | GC+PDC_Theta+Spectral | 74.8% | 74.7% |
   | 3 | XGBoost_Tuned | Combined | 74.5% | 74.3% |
   | 4 | SVM_Tuned | Combined | 74.2% | 74.2% |
   | 5 | Stacking | PDC_Theta+Spectral | 74.1% | 74.0% |
   | — | (banding) | GC saja | 55-57% | — |
   | — | (banding) | PDC_Theta saja | 58-61% | — |

   **Temuan kunci dari uji signifikansi (`significance_tests.csv`, paired t-test per fold):**
   - Semua kombinasi yang **memasukkan Spectral** signifikan lebih baik dari GC-only/PDC-only (p < 0.05, hampir semua p < 0.02) — mengonfirmasi hipotesis "fitur gabungan > fitur tunggal".
   - **Tapi**: perbedaan antar kombinasi ber-Spectral itu sendiri (Combined vs GC+PDC_Theta+Spectral vs PDC_Theta+Spectral vs Spectral-saja) **TIDAK signifikan** (p > 0.1 semua). Artinya **Spectral adalah pendorong utama** kenaikan akurasi; GC dan PDC tidak menambah kontribusi yang signifikan secara statistik di atas Spectral saja pada evaluasi ini.
   - Plateau di **~75%**, belum tembus target 80%.

### Langkah 4 (berikutnya, belum dikerjakan)
Opsi iterasi (perlu diskusi/pilihan arah dengan user, karena masing-masing scope beda):
1. **ROI aggregation** (Prefrontal/Temporal/Central) — reduksi dimensi GC/PDC yang lebih neurobiologically-informed, mungkin mengurangi noise dibanding fitur graf mentah per-channel.
2. **Class-weighting / cek balance kelas per fold** — quick check, biaya rendah.
3. **Model/meta-learner tambahan** — LightGBM, atau stacking dengan meta-learner lebih kuat dari LogisticRegression.
4. **Terima 75% sebagai plateau** dan laporkan sebagai temuan valid (naik signifikan dari baseline lama 55-61% LOSO), diskusikan keterbatasan di Bab 3.4 — mengingat temuan bahwa GC/PDC tidak menambah signifikan di atas Spectral, argumen keterbatasan konektivitas EEG untuk emosi (dibanding fitur spektral) juga jadi kontribusi ilmiah yang valid.

### Cara melanjutkan sesi berikutnya
Langkah 3 tuntas — semua output final ada di `phase_4_classification/4.1_lda` s.d. `4.7_gc_vs_pdc_comparison/output/`. Untuk Langkah 4, mulai dari salah satu opsi di atas sesuai arahan user, jangan mulai dari nol.

### Langkah 4 — ROI Aggregation (dikerjakan, SELESAI)
User memilih opsi 1 (ROI aggregation) dari 4 opsi di atas (bukan class-weighting, model tambahan,
atau menerima plateau). Target 80% juga dilonggarkan user di tengah sesi ("semaksimal mungkin,
tidak perlu 80%").

`phase_3_feature_engineering/3.4_roi_aggregation/` (ditemukan sudah dibuat dari sesi sebelumnya,
belum tercatat di sini) mengagregasi matriks GC/PDC 62x62 per-channel menjadi 6x6 antar-region
(Prefrontal, Frontal, Temporal, Central, Parietal, Occipital, dari `CHANNEL_NAMES` di `config.py`).
4 dataset ROI baru (`GC_ROI`, `PDC_ROI`, `GC+PDC_ROI`, `GC+PDC_ROI+Spectral`) di-wire ke seluruh
`4.1_lda`-`4.6_hyperparameter_tuning`, lalu `4.7` di-rerun untuk agregasi final.

**Catatan teknis penting:** 4.5 (Ensemble/Stacking) dan 4.6 (GridSearch tuning) sempat mati 2x
saat dijalankan **paralel** (2 proses `n_jobs=-1` bersamaan tampaknya memicu resource-kill di
lingkungan ini) — diperbaiki dengan (1) menambah **checkpointing per (dataset, model)** di kedua
notebook (skip pasangan yang sudah tersimpan di CSV output, auto-resume), dan (2) menjalankan job
berat **satu per satu (sekuensial)**, bukan paralel. Pelajaran ini diterapkan lagi di Phase 5.

**Hasil akhir (`4.7/output/master_comparison_table.csv`, grouped 5-fold CV per subjek):**

| Rank | Model | Dataset | Akurasi | Dimensi fitur |
|---|---|---|---|---|
| 1 | **XGBoost_Tuned** | **GC+PDC_ROI+Spectral** | **75.56% ± 4.68%** | 990 |
| 2 | Stacking | Combined (per-channel) | 75.41% ± 3.68% | 2676 |
| 3 | Voting_Soft / SVM_Tuned | GC+PDC_ROI+Spectral | ~75.0% | 990 |

**Temuan kunci (`significance_tests.csv`):** untuk **setiap** model, selisih `Combined` vs
`GC+PDC_ROI+Spectral` **tidak signifikan** (p selalu > 0.37) — ROI setara secara statistik dengan
per-channel penuh, dengan ~63% lebih sedikit fitur. ROI tanpa Spectral (GC_ROI/PDC_ROI/GC+PDC_ROI)
masih jauh di bawah (58-65%) — Spectral tetap kontributor utama (konsisten dgn Langkah 3).

**Rekomendasi Bab 3/4:** laporkan `GC+PDC_ROI+Spectral + XGBoost_Tuned (75.56%)` sebagai hasil
utama, bukan `Combined` — akurasi setara/lebih baik, dimensi jauh lebih kecil (less overfitting
risk), dan representasi region-to-region langsung relevan untuk tujuan riset #1 (komunikasi
Prefrontal/Temporal/Central).

## Phase 5 — LOSO (Leave-One-Subject-Out), di luar rencana awal ini, dikerjakan atas permintaan user (2026-07-22)

**Tujuan eksplisit dari user:** memenuhi validasi **cross-subject** yang lebih ketat/standar
literatur EEG (Phase 4 sudah subject-independent via `StratifiedGroupKFold` 5-fold grouped, tapi
LOSO — 15 fold, 1 subjek/fold — adalah gold-standard yang literatur EEG biasanya laporkan).

`phase_5_loso/code/phase_5_loso.ipynb` sebelumnya adalah stub lama pre-Langkah-1 (fitur usang,
`SelectKBest(k=50)`, tanpa Spectral/ROI, tanpa checkpointing) — **ditulis ulang dari nol**, lalu atas
permintaan user **dipecah jadi 6 notebook terpisah** (satu per sub-phase, konsisten dengan pola
Phase 4), masing-masing dengan penjelasan markdown + code output + figure ter-embed:

- `5.1_loso_cv/code/5.1_loso_cv.ipynb` — konsep LOSO + setup 5 dataset kunci + sanity check split.
- `5.2_loso_all_models/code/5.2_loso_all_models.ipynb` — LOSO CV penuh: 5 dataset (GC, PDC_Theta,
  Spectral, Combined, GC+PDC_ROI+Spectral) x 4 model dasar (LDA, SVM_Tuned, RF_Tuned,
  XGBoost_Tuned, hyperparameter dari Phase 4.6 — **tidak** re-tuning di LOSO) + Voting_Soft.
  **Checkpointing per fold** (375 total fold-run, resume otomatis jika proses mati).
  Stacking di-skip (internal CV bersarang terlalu mahal di 15 fold luar).
- `5.3_loso_analysis/code/5.3_loso_analysis.ipynb` — akurasi per-subjek (bar chart per dataset +
  ranking subjek paling sulit/mudah digeneralisasi lintas semua fitur).
- `5.4_gc_vs_pdc_loso/code/5.4_gc_vs_pdc_loso.ipynb` — paired t-test/Wilcoxon antar dataset per
  model (versi LOSO dari uji Phase 4.7) + heatmap p-value.
- `5.5_icc_analysis/code/5.5_icc_analysis.ipynb` — ICC(2,1) per fitur (GC, PDC_Theta, Spectral).
- `5.6_subject_normalization/code/5.6_subject_normalization.ipynb` — ablasi eksplisit subject-wise
  z-score normalization di bawah LOSO (yang di Phase 4 selalu baked-in, tak pernah diablasi).

**Hasil (`5.2/output/loso_results.csv`, 15-fold LOSO):**

| Rank | Model | Dataset | Akurasi LOSO |
|---|---|---|---|
| 1 | **Voting_Soft** | **GC+PDC_ROI+Spectral** | **75.70% ± 9.99%** |
| 2 | SVM_Tuned | Combined | 75.26% ± 8.52% |

Std jauh lebih besar dari Phase 4 (5-fold, ~4-5%) karena tiap fold LOSO test-nya cuma 1 subjek
(45 sampel) — wajar untuk LOSO, bukan tanda masalah.

**5.4:** Combined vs GC+PDC_ROI+Spectral tetap **tidak signifikan** di semua model (p=0.43-0.94) —
mengonfirmasi ulang temuan Phase 4 di bawah protokol paling ketat.

**5.5:** Spectral paling subject-dependent (37 fitur ICC>0.7, mean ICC 0.344) vs GC (0 fitur
ICC>0.7, mean 0.217) vs PDC_Theta (mean 0.347).

**5.6 — TEMUAN PALING PENTING Phase 5:** ablasi subject-wise normalization di bawah LOSO:

| Dataset | Tanpa normalisasi | Dengan z-score per subjek | Selisih |
|---|---|---|---|
| GC | ~54.5% | ~61.0% | +6.6pp |
| **GC+PDC_ROI+Spectral** | **~55.1%** | **~75.0%** | **+19.9pp** |

Normalisasi per-subjek (baked-in di Phase 4 tanpa pernah diuji eksplisit) ternyata krusial, bukan
penyesuaian kecil — tanpanya akurasi dataset terbaik anjlok ke level setara GC saja. Poin
metodologis penting untuk Bab 3: EEG sangat subject-specific, normalisasi per-subjek adalah
prasyarat, bukan opsi, untuk klasifikasi lintas-individu.

**Kesimpulan Phase 5:** LOSO mengonfirmasi ulang seluruh temuan Phase 4 (akurasi ~75-76%, ROI setara
Combined, plateau ~75%) di bawah protokol cross-subject gold-standard literatur EEG — syarat
"cross-subject terpenuhi" dari user tercapai — sekaligus menambah 2 kontribusi analitis baru
(per-subjek + bukti kuantitatif pentingnya normalisasi).

---

## Context

Skripsi ini membandingkan Granger Causality (GC) vs Partial Directed Coherence (PDC) sebagai fitur konektivitas untuk klasifikasi emosi EEG (dataset SEED), dengan target akurasi 70-80% (konektivitas saja) atau 75-85% (fitur gabungan + spektral), sesuai `new_data/readme.md`.

Investigasi terhadap `new_data/phase_4_classification/` menemukan dua masalah metodologis yang membuat angka akurasi terbaik saat ini (76.0%, `Voting_Soft` pada GC — lihat `4.7_gc_vs_pdc_comparison/output/master_comparison_table.csv`) **tidak reliable**:

1. **CV bocor (data leakage):** `phase_4_classification.ipynb` cell 5 memakai `StratifiedKFold(n_splits=5, shuffle=True)` — trial dari subjek yang sama bisa masuk ke train *dan* test fold sekaligus. Kolom `subjects` sudah diekstrak di `prepare_features()` tapi tidak pernah dipakai untuk grouping. Karena EEG sangat subject-specific, ini menggelembungkan akurasi.
2. **Dimensi fitur jauh melebihi jumlah sampel (p ≫ n):** `engineered_features_gc.csv` / `*_pdc_*.csv` = 675 baris × **4078 kolom fitur**; `engineered_features_combined.csv` = 675 × **28510 kolom**. `SELECT_K=1200` dipilih manual by trial-and-error dari hasil yang sudah leaky (lihat komentar "FIX 3" di cell 5). Dengan p≫n + leakage, model kemungkinan besar menghafal pola per-subjek, bukan pola emosi yang general.

Kombinasi ini berarti gap ke target 80% kemungkinan lebih besar dari yang terlihat. Keputusan yang sudah dikonfirmasi:
- **Perbaiki dulu metodologi CV** (pakai `StratifiedGroupKFold` per subjek) sebagai evaluasi utama Phase 4, supaya angka yang dikejar ke 80% valid untuk skripsi.
- **Tambahkan fitur spektral** (PSD band power + Differential Entropy) untuk menyasar skenario target tertinggi (75-85%) yang sudah didefinisikan di roadmap sendiri.

Tujuan rencana ini: menghasilkan angka akurasi Phase 4 yang jujur (subject-independent) dan strategi konkret untuk mendekatkan ke ≥80%, tanpa keluar dari lingkup "sampai Phase 4" (Phase 5 LOSO/Phase 6 DL tetap di luar scope, tapi metodologi grouped-CV di Phase 4 dibuat setara dengan LOSO agar hasilnya transferable).

**Catatan lingkup:** semua file baru (kode, notebook, CSV, dokumen) dari rencana ini dibuat di dalam `D:\Skripsi\new_data\` — tidak menyentuh `code_skripsi/`, `code_skripsi_new/`, atau folder lama lainnya.

## Langkah 1 — Fitur Spektral Baru (pelengkap GC/PDC) ✅ SELESAI

Buat ekstraksi fitur spektral dari sinyal EEG hasil preprocessing (`00_preprocessing/output/preprocessed_eeg/subject_XX/session_YYYYMMDD/trial_XX.npy`, shape `(62, timepoints)`, fs=100Hz), paralel dengan `01_granger_causality` dan `02_pdc`:

- Folder baru: `new_data/03_spectral_features/code/03_spectral_features.ipynb` (notebook, konsisten dengan pola phase_1–phase_6 lain di `new_data/` — bukan file `.py` terpisah).
- Per trial, per channel (62), hitung untuk 5 band (delta/theta/alpha/beta/gamma, konsisten dengan band PDC yang sudah ada):
  - **Band power** (Welch PSD, `scipy.signal.welch`)
  - **Differential Entropy** (`0.5 * log(2*pi*e*variance)` per band — fitur andalan paper Zheng & Lu 2015 pada dataset SEED yang sama)
- Output: CSV per subject/session/trial dengan kolom kunci yang **sama persis** dengan CSV lain (`subject_id, subject_num, session, trial, class, class_label`) supaya bisa di-join langsung — pola ini sudah dipakai konsisten di `phase_1_data_structuring/csv/*.csv`.
- Tambahkan agregasi ke `phase_1_data_structuring` (atau langsung ke `phase_3_feature_engineering`) sebagai `cleaned_spectral_features.csv`, lalu buat `engineered_features_spectral.csv` dan `engineered_features_gc_pdc_spectral.csv` (kombinasi) di `phase_3_feature_engineering/code/phase_3_feature_engineering.ipynb`.

## Langkah 2 — Perbaiki Dimensionalitas Fitur (Phase 3) ✅ SELESAI

**Hasil:** `engineered_features_gc.csv` 4078→**296 kolom**, `engineered_features_combined.csv` 28510→**2676 kolom** (edges mentah dibuang, disimpan terpisah di `engineered_features_full_*.csv` untuk eksperimen opsional). Spectral (Langkah 1) sudah masuk ke `Combined`.

Re-run `3.3_informativeness_comparison` (baseline RF, `StratifiedGroupKFold` per subjek — evaluasi jujur) dengan fitur reduced + Spectral:

| Method | #Fitur (reduced) | Mean F-Score | CV Acc (grouped, RF tunggal) |
|---|---|---|---|
| GC | 261 | 8.72 | 53.8% ± 7.8% |
| PDC Delta | 175 | 4.25 | 47.6% ± 3.4% |
| PDC Theta | 174 | 4.30 | 45.0% ± 5.4% |
| PDC Alpha | 175 | 4.31 | 49.3% ± 3.7% |
| PDC Beta | 178 | 3.91 | 49.0% ± 6.2% |
| PDC Gamma | 185 | 3.63 | 48.6% ± 6.1% |
| PDC Broadband | 181 | 3.97 | 47.4% ± 6.4% |
| **Spectral (baru)** | 382 | **8.01** | 53.3% ± 2.5% |

**Catatan penting:** ini baseline RF per **jenis fitur sendiri-sendiri** (bukan gabungan), jadi belum representatif untuk target 80% — itu baru terjawab di Langkah 3 (Phase 4: kombinasi GC+PDC+Spectral + SVM/Ensemble + tuning, dengan CV yang sama-sama grouped). Yang penting dari langkah ini:
1. Baseline jujur GC-only ada di ~54% (mengonfirmasi 76% versi lama memang inflated akibat leakage, bukan hilang setelah reduksi dimensi saja).
2. Spectral punya fitur individu paling diskriminatif (max F-score 139 vs GC 57, PDC ~50-60) — sinyal bagus untuk kontribusi di Phase 4 nanti.

## Langkah 3 — Perbaiki Evaluasi Phase 4 (CV grouped by subject)

Di `phase_4_classification/code/phase_4_classification.ipynb` (dan sub-notebook `4.5_ensemble`, `4.6_hyperparameter_tuning`, `4.7_gc_vs_pdc_comparison` yang polanya sama):

- Ganti `StratifiedKFold` → `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` dengan `groups=subjects` (variabel `subjects` sudah tersedia dari `prepare_features()`, tinggal disambungkan ke `cross_validate(..., groups=subjects)`).
- Tambahkan dataset baru ke perbandingan: `Spectral`, `GC+Spectral`, `PDC+Spectral`, `GC+PDC+Spectral` (di samping `GC`, `PDC_<best_band>`, `Combined` yang sudah ada).
- Tambahkan langkah **subject-wise normalization** (z-score fitur per subjek sebelum masuk classifier) sebagai varian yang diuji berdampingan dengan versi tanpa normalisasi — teknik ini sudah direncanakan di `phase_5_loso/5.6_subject_normalization` tapi perlu diuji juga di Phase 4 karena berdampak besar pada variabilitas antar-subjek EEG.
- `4.6_hyperparameter_tuning`: pastikan `GridSearchCV`/`Optuna` juga pakai `cv=StratifiedGroupKFold(...)` dengan `groups`, supaya hyperparameter terbaik tidak overfit ke split yang bocor.
- Re-run seluruh notebook 4.1–4.7 untuk regenerate: `classification_results.csv`, `ensemble_results.csv`, `tuning_best_params.csv`, `master_comparison_table.csv` dengan metodologi baru.

## Langkah 4 — Evaluasi & Iterasi

- Bandingkan hasil grouped-CV baru terhadap target table di `readme.md` (70-80% konektivitas saja, 75-85% gabungan).
- Jika kombinasi terbaik (kemungkinan `GC+PDC+Spectral` + `SVM`/`Ensemble Stacking`) masih di bawah 80%:
  - Coba agregasi ROI (Prefrontal/Temporal/Central — sudah jadi tujuan penelitian di readme) sebagai reduksi dimensi yang lebih neurobiologically-informed.
  - Coba class-weighting / cek balance kelas per fold.
  - Pertimbangkan model tambahan (LightGBM) atau stacking meta-learner yang lebih kuat.
  - Jika tetap plateau, laporkan sebagai temuan valid (bukan kegagalan) — bandingkan dengan baseline lama (55-61% LOSO di pipeline sebelumnya) untuk menunjukkan peningkatan riil, dan diskusikan keterbatasan di Bab 3.4.

## File Kunci yang Akan Diubah/Dibuat (semua di dalam `new_data/`)

- **Baru:** `new_data/03_spectral_features/code/spectral_analyzer.py`, `run_spectral.py`
- `new_data/phase_1_data_structuring/code/*.ipynb` — tambah agregasi spectral CSV
- `new_data/phase_3_feature_engineering/code/phase_3_feature_engineering.ipynb` — fitur spektral, prioritas graph-metric vs edge-level, informativeness comparison
- `new_data/phase_4_classification/code/phase_4_classification.ipynb` — `StratifiedGroupKFold`, dataset baru, subject normalization
- `new_data/phase_4_classification/4.5_ensemble/code/4.5_ensemble.ipynb`
- `new_data/phase_4_classification/4.6_hyperparameter_tuning/code/4.6_hyperparameter_tuning.ipynb`
- `new_data/phase_4_classification/4.7_gc_vs_pdc_comparison/code/4.7_gc_vs_pdc_comparison.ipynb`

## Verifikasi

- Jalankan tiap notebook end-to-end (`jupyter nbconvert --to notebook --execute`) dan pastikan tidak error.
- Cek `classification_results.csv` & `master_comparison_table.csv` baru: akurasi grouped-CV harus **lebih rendah dari 76%** (itu tanda leakage sudah hilang, bukan regresi) — bandingkan tren naik/turun per kombinasi fitur, bukan angka absolut lama.
- Bandingkan akurasi `GC+PDC+Spectral` vs `GC`/`PDC` saja untuk memvalidasi hipotesis "fitur gabungan > fitur tunggal" (tujuan riset #2 di readme).
- Pastikan tidak ada subjek yang sama muncul di train dan test fold yang sama (assert manual: `set(subjects[train]) & set(subjects[test]) == set()`).
