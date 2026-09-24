# 6.1 Windowed feature extraction

Ekstraksi offline window EEG 60 detik non-overlap dari output preprocessing.
Semua kode dan hasil baru berada di tahap ini; baseline tidak diedit atau dijalankan ulang.

Jalankan dari root repo (default S01, Session 1, Trial 1):

```powershell
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/extract_windows.py
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/test_windows.py
```

Pilihan trial lain: `--subject 2 --session-index 1 --trial 1`.
Session index 1–3 dipetakan ke tanggal melalui `config.SUBJECT_SESSIONS`.
Tidak ada eksekusi otomatis seluruh dataset.

## Batch extraction, resume, dan coverage

`run_batch_extraction.py` mengenumerasi tepat 15 × 3 × 15 = 675 trial. Run `passed`
yang lengkap dan memiliki hash pipeline saat ini dilewati; run `running`/gagal,
hilang, atau tidak kompatibel dijadwalkan ulang sebagai run unik. Duplicate passed
runs tidak menggandakan evidence: run valid terbaru dipilih dan sisanya dicatat.

```powershell
# Read-only inventory; tidak mengekstrak
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/check_extraction_coverage.py
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --audit-only

# Representative pilot: S01/S02, Session 1, trials 1–3 (semua kelas)
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 2 --sessions 1 --trials 1 2 3 --workers 1 --resume --dry-run

# Full plan only (safe; no extraction)
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --workers 2 --resume --dry-run
```

Eksekusi nyata dilakukan dengan menghapus `--dry-run`. Gunakan awalnya satu worker.
Sebelum `--workers > 1`, buat dua hasil untuk trial yang sama—satu melalui single-trial
extractor dan satu melalui probe child-process terbatas—lalu bandingkan:

```powershell
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/extract_windows.py --subject 1 --session-index 1 --trial 1
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/run_batch_extraction.py --subjects 1 --sessions 1 --trials 1 --worker-probe
python -B phase_6_wearable_evaluation/6.1_windowed_feature_extraction/code/check_parallel_equivalence.py --serial-run PATH_SERIAL --worker-run PATH_WORKER --tolerance 1e-10
```

Parallel mode fail-closed bila `parallel_equivalence_report.json` tidak `passed` atau
hash pipeline berubah. Worker memaksa OpenMP/OpenBLAS/MKL/NumExpr/BLIS satu thread.
`--worker-probe` hanya mengizinkan tepat satu trial dalam satu child process; ini
adalah determinism gate, bukan izin menjalankan batch paralel tanpa report.
Host audit melihat 8 physical/16 logical CPU dan sekitar 19,7 GiB RAM. Rekomendasi
untuk host ini adalah **2 workers**; coba maksimum 3 hanya setelah pilot menunjukkan
RAM tanpa paging dan throughput membaik. Jangan menyamakan jumlah worker dengan jumlah logical core: setiap
trial memuat VAR 62-channel dan ribuan fit GC, sehingga RAM dan bandwidth menjadi batas.

## Protokol

- Input `(62, samples)` pada `config.TARGET_FS` (100 Hz saat ini).
- Window 6.000 sampel dengan interval `[start_sample, stop_sample)`; ID mulai 1.
- Window tidak melewati batas trial. Sisa kurang dari 60 detik dibuang, tanpa padding.
- Semua metode menggunakan window dan 62 channel yang sama, mengikuti runner GC/PDC aktif.
  Konfigurasi lama memiliki komentar/subset ROI PDC yang tidak dipakai runner tersebut.
- GC memanggil `gc_analyzer.analyze_trial`: preprocessing internal per-window,
  lag maksimum dari config, minimum p across lags, dan threshold baseline.
- PDC memanggil `pdc_analyzer.analyze_trial`: VAR/AIC dan enam band termasuk broadband.
- Spektral memakai definisi `extract_trial_features` dari notebook lama melalui AST:
  hanya definisi fungsi dimuat, tanpa menjalankan loop/I/O notebook.
- Agregasi ROI memakai definisi kelompok dan fungsi dari notebook 3.4 dengan cara sama.
- Matriks PDC mentah disimpan sesuai analyzer (`target, source`). Untuk fitur ROI baru,
  transpose PDC dahulu sehingga nama `A_to_B` berarti sumber A ke tujuan B, konsisten
  dengan GC. Ini berbeda dari notebook ROI lama yang langsung mengagregasi PDC.
- Tidak ada normalisasi fitur berdasarkan seluruh subject/session dan tidak ada fitting classifier.

## Output

Setiap eksekusi membuat direktori unik di
`output/subject_XX/session_DATE/trial_XX/run_TIMESTAMP/`:

- `windowed_features.csv`: satu baris/window, 620 fitur spektral + 50 GC ROI +
  300 PDC ROI = 970 fitur, ditambah 12 kolom identitas/batas/label.
- `window_manifest.csv`: identitas window, berkas matriks, density GC, order VAR, durasi komputasi.
- `timing_report.json`: load/preprocess setup, spectral, GC, PDC, ROI aggregation,
  save, total, dan timing per window.
- `window_NNN.npz`: GC raw, p-values, thresholded, dan enam matriks PDC (62×62).
- `sanity_report.json`: status, SHA256 input dan sumber pipeline, parameter, channel,
  orientasi matriks, jumlah window, dan sisa sampel.

CSV disimpan setelah setiap window lengkap; hanya status `passed` menandai run lengkap.
Run terputus tetap `running` dan tidak dipakai ulang otomatis.
Kolom identitas/batas, kelas, dan metadata manifest harus dikecualikan dari input classifier.
Split evaluasi selanjutnya wajib mempertahankan grouping subject/session/trial yang relevan;
window dari trial sama tidak boleh diacak lintas train/test.

Sanity case S01 / 20131027 / trial 01: 23.501 sampel (235,01 detik),
3 window penuh dan 5.501 sampel (55,01 detik) dibuang.
Pengecekan mencakup bentuk/finite input dan output, 970 fitur, rentang PDC/p-values,
diagonal GC, dan output GC yang tidak seluruhnya nol. Analyzer GC baseline menangkap
error pasangan secara diam-diam; pemeriksaan ini tidak membuktikan semua pasangan fit sukses.

Data sudah difilter pada tahap preprocessing lama. Hasil ini merupakan simulasi window
offline, belum bukti preprocessing kausal atau performa wearable real-time.

Batch output menyimpan `batch_plan.csv`, `batch_report.json`, dan
`batch_timing_summary.csv`. Coverage dashboard menyimpan `coverage_dashboard.csv`
serta `coverage_summary.json` dengan passed/failed/missing/incompatible dan agregat
per subject/session.

## Runtime dan audit density GC

Sanity S01/S1/T1 membutuhkan 1.450,1 detik untuk tiga window. Implementasi aktif
melakukan 62×61=3.782 arah GC per window dan `statsmodels.grangercausalitytests`
mengevaluasi lag 1–10 untuk setiap arah; ini bottleneck dominan yang dipertahankan
karena mengubahnya dapat mengubah hasil ilmiah. PDC VAR/AIC adalah beban berikutnya;
spectral/ROI/I/O diprofilkan terpisah agar keputusan berbasis measurement.

Setiap window kini mencatat raw nonzero density, thresholded density, distribusi
p-value, jumlah p<alpha, dan jumlah edge ditolak FDR. Read-only audit artefak sanity
mereproduksi matriks tersimpan secara tepat dan mendapat mean density 0,904812.
Dokumentasi lama menyebut target <30% dan output percentile lama sekitar 0,1002.
Perbedaan ini konsisten dengan metode aktif: minimum p-value dipilih dari 10 lag
sebelum BH-FDR. Tidak ada threshold yang diubah. Ini harus dilaporkan/review secara
ilmiah, bukan “diperbaiki” diam-diam.

Dependensi: numpy, scipy, pandas, statsmodels, threadpoolctl (lingkungan baseline).
