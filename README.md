🔬 Eksperimen SML — Dian Fatonah
Submission Dicoding | Membangun Sistem Machine Learning
Klasifikasi Sentimen IMDB Reviews menggunakan BiLSTM + MLflow Tracking

📋 Deskripsi
Repository ini berisi proyek klasifikasi sentimen ulasan film IMDB Reviews menggunakan model BiLSTM dengan MLflow Tracking. Mencakup preprocessing data teks hingga hyperparameter tuning.

🔗 Link GitHub: github.com/Dianftnh/Dicoding-Eksperimen_SML
🗂️ Struktur Folder
Dicoding-Eksperimen_SML/
│
├── dataset_raw/
│   ├── imdb_reviews_raw.csv              # Dataset mentah
│   └── imdb_reviews_raw.xlsx             # Dataset mentah (excel)
│
├── preprocessing/
│   ├── Eksperimen_Dian_Fatonah.ipynb     # Notebook EDA & preprocessing
│   ├── X_train.npy, X_val.npy, X_test.npy   # Data siap latih
│   ├── y_train.npy, y_val.npy, y_test.npy   # Label
│   ├── label_encoder.pkl, tokenizer.pkl     # Encoder & tokenizer
│   └── *.png                              # Visualisasi EDA
│
├── Membangun_model/
│   ├── modelling.py                       # Basic MLflow autolog
│   ├── modelling_tuning.py                # Hyperparameter tuning + manual logging
│   ├── Training_Modelling.ipynb           # Notebook Colab training
│   ├── dataset_preprocessing/             # Data preprocessing (copy)
│   ├── requirements.txt                   # Dependencies
│   ├── screenshoot_dashboard.jpg          # Screenshot MLflow dashboard
│   └── screenshoot_artifak.jpg            # Screenshot MLflow artifacts
│
├── .workflow/
│   └── preprocessing.yml                  # CI preprocessing
│
├── requirements.txt
├── .gitignore
└── README.md
📊 Dataset
Properti	Detail
Sumber	keras.datasets.imdb
Jumlah Data	50.000 ulasan film
Label	0 = Negatif, 1 = Positif
Tipe	Teks bahasa Inggris
Preprocessing	Cleaning → Tokenisasi → Padding → Label Encoding → Train/Val/Test split
🧪 Notebook Preprocessing (Eksperimen_Dian_Fatonah.ipynb)
Notebook ini mencakup seluruh tahap preprocessing data secara interaktif:

Eksplorasi Data — wordcloud, distribusi kelas, panjang teks
Cleaning Teks — lowercase, hapus HTML/URL/punctuation
Tokenisasi & Padding — konversi teks ke sequence
Label Encoding — encode label sentimen
Train/Val/Test Split — pembagian dataset

⚙️ Modelling (modelling.py & modelling_tuning.py)
Hyperparameter Tuning — 3 kombinasi parameter
Manual Logging — logging metrics ke MLflow (tanpa autolog)
Artefak — model, confusion matrix, classification report

Alur Pipeline
Dataset → Preprocessing → Training (3 tuning) → Evaluasi → MLflow Tracking
Tahapan Modelling
Langkah	Fungsi	Metode
1. Load Data	load_data()	numpy.load()
2. Build Model	build_model()	BiLSTM (Embedding → BiLSTM → Dense)
3. Training	train_and_log()	EarlyStopping + ReduceLROnPlateau
4. Evaluasi	evaluate()	Accuracy, Precision, Recall, F1
5. Logging	mlflow.log_*	Manual logging params + metrics + artifacts
Output
Membangun_model/model_output/
├── model_bilstm_tuning_1.keras
├── model_bilstm_tuning_2.keras
├── model_bilstm_tuning_3.keras
├── classification_report_bilstm_tuning_1.txt
├── confusion_matrix_bilstm_tuning_1.png
└── confusion_matrix_best.png
🚀 Cara Menjalankan
Prasyarat
pip install -r Membangun_model/requirements.txt
Menjalankan Training di Colab
Buka Membangun_model/Training_Modelling.ipynb di Google Colab, jalankan seluruh cell:
- Mount Google Drive
- Clone repo dari GitHub
- Install dependencies
- Load data preprocessing dari Drive
- Hyperparameter tuning (3 kombinasi) dengan manual logging
- Evaluasi model terbaik
- Download hasil

Menampilkan MLflow UI
setelah training selesai, jalankan:
!mlflow ui --host 0.0.0.0 --port 5000 &
from pyngrok import ngrok
ngrok.set_auth_token("TOKEN_ANDA")
url = ngrok.connect(5000)
print(url)

Contoh output MLflow:
[2026-07-30] INFO - [bilstm_tuning_1] Test Acc: 0.8534, F1: 0.8532
[2026-07-30] INFO - [bilstm_tuning_2] Test Acc: 0.8612, F1: 0.8610
[2026-07-30] INFO - [bilstm_tuning_3] Test Acc: 0.8578, F1: 0.8576
[2026-07-30] INFO - Best run: bilstm_tuning_2 with test accuracy: 0.8612
🔗 Keterkaitan dengan Folder Lain
preprocessing/   ──→   Membangun_model/dataset_preprocessing/
(Data preprocessing)      (Digunakan untuk training model)

Data hasil preprocessing digunakan langsung oleh Membangun_model/ sebagai input training model BiLSTM.
👤 Author
Info	Detail
Nama	Dian Fatonah
Program	Dicoding — Membangun Sistem Machine Learning
Dataset	IMDB Reviews (Sentiment Analysis)
Model	BiLSTM + MLflow Tracking
Level	Skilled/Menengah (manual logging + hyperparameter tuning)
