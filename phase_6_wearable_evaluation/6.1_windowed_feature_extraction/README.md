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

Dependensi: numpy, scipy, pandas, statsmodels, threadpoolctl (lingkungan baseline).
