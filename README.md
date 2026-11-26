# Dry Bean Classification Submission Package

Paket bahan pengumpulan untuk tugas Matematika Diskrit: perbandingan Decision Tree dan Random Forest pada Dry Bean Dataset.

## Struktur paket

- `Dry_Bean_Classification_Colab.ipynb`: notebook utama untuk Google Colab.
- `Dry_Bean_Classification_Analysis.ipynb`: notebook generik untuk Jupyter.
- `data/`: CSV asli dan arsip unduhan sumber.
- `outputs/`: dataset bersih, EDA, metrik, classification report, tabel perbandingan, dan visualisasi.
- `models/`: model baseline dan tuned dalam format Joblib.
- `evidence/`: training/testing set, hasil cross-validation, prediksi, confusion matrix berbentuk CSV, aturan pohon, metadata, log, dan verifikasi.
- `source_code/`: alternatif kode Python `.py`.
- `data_dictionary.csv`, `references.md`, `requirements.txt`: dokumentasi dan dependensi.
- `PANDUAN_PENGUMPULAN.md`: penjelasan file inti dan file pendukung.
- `MANIFEST_SHA256.csv`: daftar file dan checksum untuk pemeriksaan integritas.

## Ringkasan eksperimen

- Dataset mentah: 13.611 baris, 16 fitur prediktor, 1 target.
- Target: `Class`, terdiri dari 7 varietas.
- Missing values: 0 pada audit awal.
- Duplikasi eksak yang dihapus: 68 baris.
- Dataset bersih: 13.543 baris.
- Split: 80% training (10.834 baris) dan 20% testing (2.709 baris), stratified, `random_state=42`.
- Model: Decision Tree dan Random Forest.
- Tuning: 5-fold Stratified Cross-Validation; Macro F1 sebagai skor pemilihan.
- Kandidat yang dievaluasi: 8 Decision Tree dan 7 Random Forest.
- Testing set tidak digunakan untuk memilih hyperparameter.

## Hasil pengujian final

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Decision Tree (tuned) | 89,48% | 90,81% |
| Random Forest (tuned) | 91,77% | 92,96% |

Random Forest memiliki Macro F1 testing tertinggi pada konfigurasi yang diuji. Hasil ini berlaku untuk dataset, preprocessing, split, random state, dan kandidat parameter yang dievaluasi.

## Menggunakan Google Colab

1. Buka https://colab.research.google.com/.
2. Pilih **File -> Upload notebook**.
3. Unggah `Dry_Bean_Classification_Colab.ipynb`.
4. Untuk melihat hasil yang sudah tertanam, cukup buka notebook.
5. Untuk mereproduksi, pilih **Runtime -> Run all**. Sel bootstrap akan meminta satu kali unggahan ZIP final; pilih `Dry_Bean_Classification_Submission_FINAL.zip`. Notebook mengekstrak ZIP dan mendeteksi folder proyek secara otomatis.

Notebook generik dapat digunakan di Jupyter lokal. Jalankan dari root folder proyek agar path `data/`, `outputs/`, `models/`, dan `evidence/` terdeteksi.

## Catatan metodologi dan batasan

- Macro F1 dipilih sebagai metrik utama karena distribusi kelas tidak sama.
- Duplikasi eksak dihapus pada level seluruh baris. Karena dataset tidak menyediakan ID spesimen, tidak dapat dipastikan apakah baris identik merupakan entri berulang atau spesimen berbeda dengan pengukuran yang sama.
- Feature importance berbasis impurity menunjukkan kontribusi relatif dalam model, bukan hubungan sebab-akibat.
- `.joblib` bergantung pada kompatibilitas versi Python dan scikit-learn. Lihat `requirements.txt`.
- Laporan PDF belum disertakan sesuai keputusan kelompok untuk menyiapkannya pada tahap berikutnya.
