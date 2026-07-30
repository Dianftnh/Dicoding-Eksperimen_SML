# Eksperimen SML — Dian Fatonah

> **Submission Dicoding | Membangun Sistem Machine Learning**  
> Klasifikasi Sentimen IMDB Reviews menggunakan BiLSTM + MLflow Tracking

---

## Deskripsi

Repository ini berisi proyek **Membangun Sistem Machine Learning** untuk klasifikasi sentimen ulasan film IMDB. Proyek mencakup preprocessing data teks (tokenisasi, padding, encoding label) hingga pembangunan model BiLSTM dengan hyperparameter tuning menggunakan MLflow Tracking.

- **Dataset**: IMDB Reviews (`keras.datasets.imdb`)
- **Model**: Bidirectional LSTM (Embedding → BiLSTM → Dense)
- **Tracking**: MLflow lokal (luring)
- **Level**: Skilled/Menengah — manual logging + hyperparameter tuning

---

## Struktur Folder

```
.
├── .workflow/
│   └── preprocessing.yml                          # CI preprocessing (GitHub Actions)
├── dataset_raw/
│   └── imdb_reviews_raw.csv / .xlsx               # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Dian_Fatonah.ipynb              # Notebook preprocessing & EDA
│   ├── X_train.npy, X_val.npy, X_test.npy         # Data siap latih
│   ├── y_train.npy, y_val.npy, y_test.npy         # Label
│   ├── label_encoder.pkl, tokenizer.pkl           # Encoder & tokenizer
│   └── *.png (class_distribution, wordcloud, dll)  # Visualisasi EDA
├── Membangun_model/
│   ├── modelling.py                                # Basic MLflow autolog
│   ├── modelling_tuning.py                         # Hyperparameter tuning + manual logging
│   ├── Training_Modelling.ipynb                    # Notebook Colab untuk training
│   ├── dataset_preprocessing/                      # Copy data preprocessing
│   ├── requirements.txt                            # Dependencies
│   ├── DagsHub.txt                                 # (tidak dipakai untuk Menengah)
│   ├── screenshoot_dashboard.jpg                   # Screenshot MLflow dashboard
│   └── screenshoot_artifak.jpg                     # Screenshot MLflow artifacts
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset: IMDB Reviews

| Properti | Detail |
|---|---|
| **Sumber** | `keras.datasets.imdb` |
| **Jumlah Data** | 50.000 ulasan film |
| **Label** | 0 = Negatif, 1 = Positif |
| **Tipe** | Teks bahasa Inggris |
| **Preprocessing** | Tokenisasi → Padding → Label Encoding → Train/Val/Test split |

---

## Alur Kerja

```
Lokal (coding)
    ↓ push
GitHub
    ↓ clone
Google Colab — mount Google Drive (data preprocessing) → training → MLflow tracking
    ↓
Screenshot dashboard & artifacts → screenshoot_*.jpg
```

### 1. Preprocessing (Kriteria 1)
Jalankan `preprocessing/Eksperimen_Dian_Fatonah.ipynb` di Colab:
- Eksplorasi data & visualisasi (wordcloud, distribusi kelas, panjang teks)
- Tokenisasi & padding sequence
- Label encoding
- Train/Val/Test split
- Simpan output (.npy, .pkl, .png) ke Google Drive

### 2. Training Model (Kriteria 2)
Buka `Membangun_model/Training_Modelling.ipynb` di Colab:
- Mount Google Drive
- Clone repo dari GitHub
- Install dependencies
- Load data preprocessing dari Drive
- Hyperparameter tuning (3 kombinasi) dengan manual logging ke MLflow
- Evaluasi model terbaik
- Download hasil (model.keras, confusion matrix, classification report)

### 3. Screenshot
Setelah training selesai, jalankan MLflow UI via ngrok:
```python
!mlflow ui --host 0.0.0.0 --port 5000 &
from pyngrok import ngrok
ngrok.set_auth_token("TOKEN_ANDA")
url = ngrok.connect(5000)
print(url)
```
- **Dashboard**: Halaman utama MLflow (daftar 3 runs dan metrics)
- **Artifacts**: Klik run → tab Artifacts (model, classification report, confusion matrix)

---

## Hyperparameter Tuning

3 kombinasi hyperparameter yang dijalankan:

| Run | Embedding Dim | LSTM Units | Learning Rate |
|---|---|---|---|
| `bilstm_tuning_1` | 128 | 64 | 1e-3 |
| `bilstm_tuning_2` | 128 | 128 | 1e-3 |
| `bilstm_tuning_3` | 256 | 64 | 5e-4 |

Semua run di-track dengan **manual logging** ke MLflow lokal:
- Parameters: embedding_dim, lstm_units, learning_rate
- Metrics: val_accuracy, val_loss, test_accuracy, precision, recall, f1
- Artifacts: model, confusion matrix (.png), classification report (.txt)

---

## Author

| Info | Detail |
|---|---|
| **Nama** | Dian Fatonah |
| **Program** | Dicoding — Membangun Sistem Machine Learning |
| **Dataset** | IMDB Reviews (Sentiment Analysis) |
| **Model** | BiLSTM + MLflow Tracking |
