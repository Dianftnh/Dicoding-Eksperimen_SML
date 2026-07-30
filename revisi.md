🔬 Eksperimen SML — Ivan Alif Hadrian
Submission Dicoding | Membangun Sistem Machine Learning
Tahap 1: Eksplorasi Data & Automated Preprocessing Pipeline

📋 Deskripsi
Repository ini merupakan fase pertama dari proyek Machine Learning end-to-end yang berfokus pada eksplorasi data (EDA) dan preprocessing otomatis terhadap dataset Heart Disease. Eksperimen ini menjadi fondasi untuk seluruh pipeline ML yang dikembangkan di tahap berikutnya.

🔗 Link GitHub: github.com/Ivalhad/EksperimenSMLIvanAlifHadrian
🗂️ Struktur Folder
Eksperimen_SML_IvanAlifHadrian/
│
├── heart_disease_raw/
│   └── heart.csv                          # Dataset mentah Heart Disease (UCI)
│
└── preprocessing/
    ├── Eksperimen_SML_IvanAlifHadrian.ipynb  # Notebook EDA & eksplorasi
    ├── automateIvanAlifHadrian.py            # Script preprocessing otomatis
    └── heart_disease_preprocessing/
        ├── train.csv                         # Data training (hasil preprocessing, 80%)
        └── test.csv                          # Data testing (hasil preprocessing, 20%)
📊 Dataset
Properti	Detail
Nama Dataset	Heart Disease (UCI Repository)
File	heart_disease_raw/heart.csv
Ukuran	±303 baris, 14 kolom
Target	target (0 = tidak sakit jantung, 1 = sakit jantung)
Fitur Numerik	age, trestbps, chol, thalach, oldpeak
Fitur Kategorikal	sex, cp, fbs, restecg, exang, slope, ca, thal
🧪 Notebook EDA (Eksperimen_SML_IvanAlifHadrian.ipynb)
Notebook ini mencakup seluruh tahap eksplorasi data secara interaktif:

Statistik Deskriptif — distribusi, mean, median, std per fitur
Visualisasi Data — histogram, boxplot, heatmap korelasi
Analisis Missing Values & Duplikat — identifikasi kualitas data
Analisis Outlier — deteksi menggunakan metode IQR
Distribusi Target — analisis keseimbangan kelas
⚙️ Automated Preprocessing Pipeline (automateIvanAlifHadrian.py)
Script Python yang mengotomatisasi seluruh proses preprocessing dari raw data hingga data siap training.

Alur Pipeline
heart.csv  →  [LOAD]  →  [DEDUP]  →  [MISSING]  →  [OUTLIER]  →  [ENCODE]  →  [SCALE]  →  [SPLIT & SAVE]
Tahapan Preprocessing
Langkah	Fungsi	Metode
1. Load Data	load_data()	pd.read_csv()
2. Hapus Duplikat	remove_duplicates()	drop_duplicates()
3. Handle Missing Values	handle_missing_values()	Median (numerik), Mode (kategorikal)
4. Handle Outlier	handle_outliers()	IQR Clipping (clip ke batas bawah/atas)
5. Encoding Kategorikal	encode_categorical()	Integer casting
6. Normalisasi Fitur	normalize_features()	StandardScaler (Z-score)
7. Split & Save	split_and_save()	80% train / 20% test (stratified)
Output
heart_disease_preprocessing/
├── train.csv   # ±242 baris (80% stratified)
└── test.csv    # ±61 baris (20% stratified)
🚀 Cara Menjalankan
Prasyarat
pip install pandas numpy scikit-learn
Menjalankan Preprocessing Otomatis
cd preprocessing
python automateIvanAlifHadrian.py
Contoh output:

==================================================
AUTOMATE PREPROCESSING PIPELINE
==================================================
[LOAD]    Dataset dimuat: 303 baris, 14 kolom
[DEDUP]   Duplikat dihapus: 1 baris | Sisa: 302 baris
[MISSING] Tidak ada missing values.
[OUTLIER] age: 0 outlier di-clip ke [29.00, 77.00]
[OUTLIER] chol: 3 outlier di-clip ke [149.50, 360.50]
[ENCODE]  Kolom kategorikal diproses: ['sex', 'cp', ...]
[SCALE]   StandardScaler diterapkan pada: ['age', 'trestbps', ...]
[SAVE]    train.csv → 242 baris
[SAVE]    test.csv  → 61 baris
[SAVE]    Output disimpan di: ./heart_disease_preprocessing
==================================================
PIPELINE SELESAI
==================================================
🔗 Keterkaitan dengan Folder Lain
Eksperimen_SML_IvanAlifHadrian   ──→   SMSML_IvanAlifHadrian   ──→   Workflow-CI
     (EDA & Preprocessing)               (Modelling & Serving)         (CI Otomatis)

heart_disease_preprocessing/
   ├── train.csv  ──────────────────────────────────────────────→ Digunakan untuk training model
   └── test.csv   ──────────────────────────────────────────────→ Digunakan untuk evaluasi & monitoring
Data hasil preprocessing (train.csv & test.csv) digunakan langsung oleh folder SMSML_IvanAlifHadrian sebagai input training model dan oleh Workflow-CI dalam pipeline CI otomatis.

👤 Author
Info	Detail
Nama	Ivan Alif Hadrian
Program	Dicoding — Membangun Sistem Machine Learning
Tahap	1 — Eksplorasi Data & Preprocessing