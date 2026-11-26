# Panduan file pengumpulan

## File inti yang direkomendasikan

1. `Dry_Bean_Classification_Colab.ipynb` — notebook utama untuk Google Colab, berisi kode, tahapan analisis, output, tabel, dan grafik.
2. `data/Dry_Bean.csv` — dataset sumber.
3. `outputs/cleaned_dry_bean.csv` — dataset setelah audit dan deduplikasi.
4. `evidence/training_set.csv` dan `evidence/testing_set.csv` — bukti split 80:20.
5. `models/` — model Decision Tree dan Random Forest, baseline maupun tuned.
6. `outputs/*.png` — visualisasi EDA dan evaluasi.
7. `outputs/*.csv` — metrik, classification report, feature importance, statistik, dan hasil tuning.
8. `evidence/` — prediksi, hasil cross-validation, confusion matrix, metadata, log, aturan Decision Tree, dan verifikasi artefak.

## File tambahan

- `Dry_Bean_Classification_Analysis.ipynb`: notebook generik untuk Jupyter.
- `source_code/Dry_Bean_Classification_Analysis.py`: alternatif source code Python.
- `requirements.txt`: dependensi Python.
- `references.md` dan `data_dictionary.csv`: referensi dan kamus data.
- `data/original_upload.zip`: arsip unduhan awal untuk jejak sumber.
- `MANIFEST_SHA256.csv`: checksum setiap file untuk pemeriksaan integritas.

## Apakah semua file wajib dikumpulkan?

Catatan dosen yang tersedia menyebut dataset Kaggle, Supervised Learning, Decision Tree, Random Forest, serta training dan testing dengan data training lebih besar. Catatan tersebut belum menetapkan format file pengumpulan secara eksplisit.

Karena itu, ZIP ini menyiapkan bukti yang lengkap. Jika dosen atau LMS hanya meminta beberapa format tertentu, instruksi resmi tersebut tetap menjadi prioritas. Untuk membuktikan proses training, notebook `.ipynb` merupakan file utama; CSV saja tidak cukup menjelaskan kode dan langkah eksperimen.

PDF laporan sengaja belum dimasukkan dan akan dibuat pada tahap berikutnya.
