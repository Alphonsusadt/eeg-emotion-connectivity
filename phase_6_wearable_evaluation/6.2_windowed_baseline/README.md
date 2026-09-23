# Phase 6.2: Windowed LOSO baseline (zero calibration)

Mengukur klasifikasi subject-independent pada window 60 detik dari Phase 6.1.
Phase 4/5 dan hasil 6.1 hanya dibaca. Semua hasil baru disimpan di tahap ini.

## Input dan coverage

Runner mencari `6.1_windowed_feature_extraction/output/subject_*/session_*/trial_*/run_*/sanity_report.json`.
Hanya status `passed` diterima. Untuk rerun trial yang sama, pilih nama timestamp run
`passed` terbaru secara leksikografis; run lama dicatat sebagai superseded, tidak digabung dua kali.
Run non-passed diabaikan dan dicatat. Run passed yang rusak ditolak, bukan dilewati diam-diam.
Semua CSV terpilih digabung berdasarkan baris; tidak ada averaging lintas window.

Validasi meliputi identitas subject/session/trial dan label terhadap config, schema persis,
nilai fitur finite, urutan ID window, interval sampel tanpa gap/overlap, panjang 60 detik,
jumlah window penuh dari input_shape, sisa sampel, dan jumlah completed_windows.
Hash sumber pipeline serta parameter/orientasi/normalisasi harus konsisten antar run.
Coverage lengkap adalah **45 trial per subject** (3 sesi × 15 trial sesuai config),
bukan sekadar satu CSV untuk setiap subject. Report mencantumkan trial yang masih kurang.

- Default maupun `--require-all-subjects`: menolak fitting/evaluasi final sebelum 15 subject lengkap.
- `--pilot`: menerima seluruh window passed yang tersedia, termasuk subject parsial;
  hasil selalu berlabel pilot. Minimal dua subject dan semua tiga kelas pada setiap training fold.
- `--audit-only`: validasi input/coverage tanpa fit classifier; dapat digunakan saat baru satu subject tersedia.
- `--pilot` dan `--require-all-subjects` saling eksklusif.

## Exact feature sets

| Dataset | Kolom | Jumlah |
|---|---|---:|
| Spectral | `spec_{power,de}_{delta,theta,alpha,beta,gamma}_{channel}`; 62 channel config | 620 |
| Connectivity | `gc_roi_*` dan `pdc_{delta,theta,alpha,beta,gamma,broadband}_roi_*` | 350 |
| Fusion | gabungan Spectral dan Connectivity pada baris/window yang sama | 970 |

Setiap prefix konektivitas berisi 50 fitur: 30 directed inter-region edges,
6 intra-region strengths, 6 in-strength, 6 out-strength, mean inter-strength,
dan mean intra-strength. Region: Prefrontal, Frontal, Temporal, Central, Parietal,
Occipital. Nama persis dihasilkan oleh `feature_sets()` dan disimpan dalam run_report.
Orientasi fitur ROI mengikuti output 6.1 (source → target).

Metadata berikut tidak pernah masuk X:
`subject_id, subject_num, session, session_index, trial, class, class_label,
window_id, start_sample, stop_sample, start_seconds, stop_seconds`.
Kolom tambahan tak dikenal ditolak. Tidak ada pemilihan band atau penyaringan variansi
berdasarkan seluruh dataset. Schema tetap, termasuk fitur konstan.

## Model dan pencegahan leakage

`LeaveOneGroupOut(groups=subject_num)` menahan satu subject penuh. Seluruh sesi, trial,
dan window subject tersebut hanya masuk test. Semua fold diperiksa sebelum fitting:
subject train/test terpisah, test hanya satu subject, training memiliki tiga kelas.
Pipeline baru dibuat per fold:

`StandardScaler -> SelectKBest(f_classif, k=min(1200, n_features)) -> SVC(RBF)`

Scaler dan selector hanya `fit(X_train, y_train)`. Karena semua feature set ≤1200,
selector mempertahankan seluruh fitur, mengikuti aturan k Phase 5. Tidak ada tuning baru.
Prediksi memakai `SVC.predict`, seperti ablasi Phase 5.6; `probability=False` karena
tidak perlu voting/probability calibration. Label tetap 0=negative, 1=neutral, 2=positive.
Macro F1 dan confusion matrix selalu memakai urutan tiga label tersebut, zero_division=0.

Parameter fixed dibaca dari satu row SVM_Tuned per dataset di
`phase_4_classification/4.6_hyperparameter_tuning/output/tuning_best_params.csv`:

| Dataset baru | Row historis | C saat implementasi | gamma |
|---|---|---:|---|
| Spectral | Spectral | 1 | scale |
| Connectivity | GC+PDC_ROI | 1 | 0.001 |
| Fusion | GC+PDC_ROI+Spectral | 1 | scale |

Jika row hilang/ambigu, runner menolak; tidak ada fallback diam-diam. Pemetaan, nilai,
dan SHA256 CSV parameter dicatat sebelum fitting. `gamma=scale` dihitung SVC hanya
dari training fold. Parameter kompatibel secara API dan dipakai sebagai transfer
historis; fitur window mentah tidak identik distribusinya dengan fitur trial/normalisasi lama.
Tuning historis memakai cohort SEED yang sama, sehingga hasil ini **bukan estimasi
dengan model selection independen/nested**. Tidak ada akses label test untuk pemilihan
parameter baru dalam 6.2, tetapi penggunaan historis tersebut tetap perlu dilaporkan.

## Zero calibration versus Phase 5

Phase 5 menormalkan setiap subject menggunakan mean/std seluruh trial subject itu,
termasuk subject held-out. Phase 6.2 tidak menghitung statistik subject/session test
dan tidak memakai full-subject z-score, baik untuk train maupun test. Test hanya
ditransformasi oleh scaler yang dipelajari dari subject training. Ini memisahkan baseline
zero-calibration dari eksperimen calibration 6.3. Preprocessing internal per-window
GC yang sudah ada pada 6.1 tetap bagian ekstraksi fitur, bukan kalibrasi antar-window.
Input masih berasal dari preprocessing offline lama, bukan validasi streaming kausal.

## Menjalankan

Dari root repo:

```powershell
# Pemeriksaan saja; tidak melatih classifier
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --audit-only
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/test_windowed_loso.py

# Hanya dijalankan ketika eksperimen diminta
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --pilot
python -B phase_6_wearable_evaluation/6.2_windowed_baseline/code/run_windowed_loso.py --require-all-subjects
```

Dependensi: numpy, pandas, scikit-learn, matplotlib, dan config repo.
Unit tests memakai data sintetis kecil dan direktori temporer di output/6.2;
tidak menjalankan ekstraksi EEG atau klasifikasi dataset nyata.

## Output

Run unik mencegah tercampurnya hasil pilot/final atau hasil lama:
`output/{audit,pilot,final}/TIMESTAMP/`; gambar di `figures/{pilot,final}/TIMESTAMP/`.

- `coverage_report.json`: subject lengkap/tersedia, missing trials, sumber run dan hash CSV/report.
- `merged_windowed_features.csv`: semua window passed terpilih (evaluasi saja).
- `run_report.json`: status, mode, exact feature lists, parameter dan provenance.
- `windowed_loso_per_subject.csv`: accuracy, macro/weighted F1, train/test window count,
  subject ID, feature count, confusion matrix (baris actual, kolom predicted).
- `windowed_loso_summary.csv`: mean dan sample SD (ddof=1) per metric **antar subject,
  bobot subject sama**, bukan akurasi pooled berbobot jumlah window.
- `windowed_predictions.csv`: prediksi out-of-fold dan metadata setiap window/dataset.
- `per_subject_accuracy.png` dan `dataset_comparison.png`: accuracy per subject dan mean ± SD.
- `failure.json` jika input/evaluasi ditolak. Hanya run_report status passed merupakan hasil lengkap.

Sebelum final: perlu 675 trial passed yang kompatibel (15×3×15), setiap trial berisi
seluruh window 60 detik lengkapnya. Coverage parsial tidak boleh dilaporkan sebagai hasil final.
