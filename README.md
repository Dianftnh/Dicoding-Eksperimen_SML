# 🔬 Eksperimen SML — Dian Fatonah

**Submission Dicoding | Membangun Sistem Machine Learning**  
Tahap 1: Eksplorasi Data & Preprocessing Pipeline

---

## 📋 Deskripsi

Repository ini merupakan tahap pertama dari proyek Machine Learning yang berfokus pada eksplorasi data (EDA) dan preprocessing terhadap dataset **IMDB Reviews**.

🔗 **GitHub**: [github.com/Dianftnh/Dicoding-Eksperimen_SML](https://github.com/Dianftnh/Dicoding-Eksperimen_SML)

---

## 🗂️ Struktur Folder

```
Dicoding-Eksperimen_SML/
│
├── dataset_raw/
│   └── imdb_reviews_raw.csv              # Dataset mentah IMDB Reviews
│
├── preprocessing/
│   ├── Eksperimen_Dian_Fatonah.ipynb     # Notebook EDA & preprocessing
│   ├── X_train.npy, X_val.npy, X_test.npy    # Data siap latih
│   ├── y_train.npy, y_val.npy, y_test.npy    # Label
│   ├── label_encoder.pkl, tokenizer.pkl      # Encoder & tokenizer
│   └── *.png                              # Visualisasi EDA
│
├── .workflow/
│   └── preprocessing.yml                  # CI preprocessing (GitHub Actions)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset

| Properti | Detail |
|----------|--------|
| **Sumber** | `keras.datasets.imdb` |
| **Jumlah Data** | 50.000 ulasan film |
| **Label** | 0 = Negatif, 1 = Positif |
| **Tipe** | Teks bahasa Inggris |

---

## 🧪 Notebook Preprocessing

**File**: `preprocessing/Eksperimen_Dian_Fatonah.ipynb`

Notebook mencakup seluruh tahap preprocessing secara interaktif:

- **Eksplorasi Data** — wordcloud, distribusi kelas, panjang teks
- **Cleaning Teks** — lowercase, hapus HTML/URL/punctuation
- **Tokenisasi & Padding** — konversi teks ke sequence
- **Label Encoding** — encode label sentimen
- **Train/Val/Test Split** — pembagian dataset

### Alur Pipeline

```
Dataset → Cleaning → Tokenisasi → Padding → Encoding → Split → Save
```

### Output

```
preprocessing/
├── X_train.npy, X_val.npy, X_test.npy    # Data siap latih
├── y_train.npy, y_val.npy, y_test.npy    # Label
├── label_encoder.pkl, tokenizer.pkl      # Encoder & tokenizer
└── *.png                                  # Visualisasi EDA
```

---

## 🚀 Cara Menjalankan

### Prasyarat

```bash
pip install -r requirements.txt
```

### Menjalankan Preprocessing

Buka `preprocessing/Eksperimen_Dian_Fatonah.ipynb` di **Google Colab**, lalu jalankan seluruh cell:

1. Mount Google Drive
2. Load dataset IMDB dari `keras.datasets.imdb`
3. Eksplorasi data & visualisasi
4. Cleaning teks
5. Tokenisasi & padding
6. Label encoding
7. Train/Val/Test split
8. Simpan output (.npy, .pkl, .png) ke Google Drive

---

## 👤 Author

| Info | Detail |
|------|--------|
| **Nama** | Dian Fatonah |
| **Program** | Dicoding — Membangun Sistem Machine Learning |
| **Dataset** | IMDB Reviews (Sentiment Analysis) |
| **Tahap** | 1 — Eksplorasi Data & Preprocessing |
