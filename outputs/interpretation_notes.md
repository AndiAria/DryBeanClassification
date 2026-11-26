# Catatan interpretasi hasil eksperimen

## Hasil utama setelah tuning

- Dataset mentah berisi 13.611 observasi dan 16 fitur prediktor.
- Audit menemukan 68 baris duplikasi eksak. Semua baris duplikasi tersebut berasal dari kelas HOROZ. Setelah satu salinan dari setiap baris identik dipertahankan, dataset berisi 13.543 observasi.
- Data dibagi secara stratified menjadi 10.834 observasi training dan 2.709 observasi testing.
- Hyperparameter dipilih menggunakan 5-fold Stratified Cross-Validation pada training set dengan Macro F1.
- Pencarian mengevaluasi 8 konfigurasi Decision Tree dan 7 konfigurasi Random Forest.

## Performa testing

- Decision Tree baseline: Accuracy 89,55%; Macro F1 90,98%.
- Decision Tree tuned: Accuracy 89,48%; Macro F1 90,81%.
- Random Forest baseline: Accuracy 91,69%; Macro F1 92,90%.
- Random Forest tuned: Accuracy 91,77%; Macro F1 92,96%.

Random Forest tuned memperoleh Macro F1 testing tertinggi pada eksperimen ini. Namun, peningkatan terhadap Random Forest baseline relatif kecil, sekitar 0,06 poin persentase Macro F1. Tuning tidak meningkatkan performa testing Decision Tree. Hal ini bukan kegagalan eksperimen: parameter dipilih berdasarkan rata-rata cross-validation pada training set, sehingga hasil pada testing set yang tidak dilihat sebelumnya dapat sedikit lebih rendah.

## Parameter terbaik yang tercatat

- Decision Tree: criterion=`entropy`, max_depth=25, min_samples_split=10, min_samples_leaf=4, max_features=None, class_weight=None.
- Random Forest: n_estimators=150, criterion=`entropy`, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=`sqrt`, class_weight=None.

## Keterbatasan dan kehati-hatian

1. Sumber tidak menyediakan ID spesimen. Karena itu, tidak dapat dipastikan apakah 68 baris identik merupakan duplikasi administrasi atau spesimen berbeda dengan pengukuran yang sama. Penghapusan dilakukan untuk menghindari rekaman identik tersebar di training dan testing, tetapi keputusan ini mengubah distribusi kelas HOROZ.
2. Hasil tuning terbatas pada kandidat parameter yang diuji, bukan pencarian seluruh ruang parameter.
3. Fitur geometris memiliki korelasi yang kuat. Feature importance impurity dapat membagi kepentingan di antara fitur berkorelasi dan tidak boleh ditafsirkan sebagai kausalitas.
4. Hasil berlaku pada dataset, preprocessing, random state, dan split yang digunakan.
