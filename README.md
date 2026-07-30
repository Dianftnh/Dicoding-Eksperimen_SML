# Eksperimen SML — Dian Fatonah

Submission Dicoding — Membangun Sistem Machine Learning  
Klasifikasi Sentimen IMDB Reviews menggunakan BiLSTM + MLflow Tracking

---

## Struktur Folder

```
.
├── .workflow/
│   └── preprocessing.yml
├── dataset_raw/
│   ├── imdb_reviews_raw.csv
│   └── imdb_reviews_raw.xlsx
├── preprocessing/
│   ├── Eksperimen_Dian_Fatonah.ipynb
│   ├── X_train.npy, X_val.npy, X_test.npy
│   ├── y_train.npy, y_val.npy, y_test.npy
│   ├── label_encoder.pkl, tokenizer.pkl
│   └── *.png (class_distribution, wordcloud, text_length_distribution)
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── Training_Modelling.ipynb
│   ├── dataset_preprocessing/
│   ├── requirements.txt
│   ├── DagsHub.txt
│   ├── screenshoot_dashboard.jpg
│   └── screenshoot_artifak.jpg
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

- **Sumber**: `keras.datasets.imdb` — 50.000 ulasan film
- **Label**: 0 = Negatif, 1 = Positif
- **Preprocessing**: Cleaning → Tokenisasi → Padding → Label Encoding → Train/Val/Test split

---

## Kriteria 1 — Preprocessing

Jalankan `preprocessing/Eksperimen_Dian_Fatonah.ipynb` di Google Colab:
- EDA (wordcloud, distribusi kelas, panjang teks)
- Cleaning teks
- Tokenisasi & padding
- Label encoding
- Train/Val/Test split
- Output: .npy, .pkl, .png → simpan ke Google Drive

## Kriteria 2 — Modelling (Skilled/Menengah)

Jalankan `Membangun_model/Training_Modelling.ipynb` di Google Colab:
1. Mount Google Drive
2. Clone repo dari GitHub
3. Install dependencies
4. Load data preprocessing dari Drive
5. Hyperparameter tuning (3 kombinasi) + manual logging ke MLflow
6. Evaluasi model terbaik
7. Download hasil

**Setelah training**, untuk screenshot:
```
!mlflow ui --host 0.0.0.0 --port 5000 &
from pyngrok import ngrok
ngrok.set_auth_token("TOKEN_ANDA")
url = ngrok.connect(5000)
print(url)
```

---

## Hyperparameter Tuning

| Run | Embedding Dim | LSTM Units | Learning Rate |
|---|---|---|---|
| bilstm_tuning_1 | 128 | 64 | 1e-3 |
| bilstm_tuning_2 | 128 | 128 | 1e-3 |
| bilstm_tuning_3 | 256 | 64 | 5e-4 |

Manual logging: params, metrics (accuracy, precision, recall, f1), artifacts (model, confusion matrix, classification report).

---

Dibuat oleh **Dian Fatonah** — Dicoding 2026
