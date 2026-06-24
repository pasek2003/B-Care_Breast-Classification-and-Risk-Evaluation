<p align="center">
  <img src="bcare/bcare_streamlit_app/bcare_streamlit_app/Images/bcare_logo.png" width="220" alt="B-Care Logo">
</p>

<h1 align="center">B-Care (Breast Classification and Risk Evaluation)</h1>

<p align="center">
  Aplikasi klasifikasi kanker payudara berbasis Streamlit menggunakan model <b>Support Vector Machine (SVM)</b>, <b>Firefly Algorithm (FA)</b>, dan <b>Fisher Score Feature Selection</b>.
</p>

<p align="center">
  <a href="https://b-carebreast-classification-and-risk-evaluation-fy2nnh9chup36z.streamlit.app/" target="_blank">
    <b>🌐 Buka Aplikasi Streamlit</b>
  </a>
</p>

---

## 📌 Tentang Aplikasi

**B-Care (Breast Classification and Risk Evaluation)** adalah aplikasi berbasis web yang dibuat untuk mendemonstrasikan hasil implementasi model machine learning dalam proses klasifikasi data kanker payudara. Aplikasi ini menggunakan dataset **Breast Cancer Wisconsin Diagnostic** untuk mengklasifikasikan data ke dalam dua kelas, yaitu:

- **Benign / Jinak**
- **Malignant / Ganas**

Aplikasi ini dibuat sebagai bagian dari Tugas Akhir dengan judul:

> **Identifikasi Kanker Payudara Menggunakan Hibridisasi Support Vector Machine (SVM) Berbasis Firefly Algorithm (FA)**

> ⚠️ **Catatan Penting:**  
> B-Care hanya digunakan untuk kebutuhan akademik, demonstrasi model, dan implementasi hasil penelitian. Hasil klasifikasi dari aplikasi ini **tidak dapat dijadikan dasar diagnosis medis langsung**. Diagnosis tetap memerlukan pemeriksaan dan penilaian tenaga medis yang berwenang.

---

## 🌐 Demo Aplikasi

Aplikasi dapat diakses melalui link berikut:

🔗 **https://b-carebreast-classification-and-risk-evaluation-fy2nnh9chup36z.streamlit.app/**

---

## 🧠 Metode dan Model yang Digunakan

Penelitian ini menerapkan model **Support Vector Machine (SVM) Linear** untuk melakukan klasifikasi data kanker payudara. SVM bekerja dengan mencari hyperplane terbaik yang mampu memisahkan data ke dalam dua kelas, yaitu jinak dan ganas.

Untuk meningkatkan performa model, digunakan metode optimasi **Firefly Algorithm (FA)**. Firefly Algorithm digunakan untuk mencari kombinasi parameter SVM yang lebih optimal, yaitu:

- **Learning rate**
- **Lambda / parameter regularisasi**
- **C / cost parameter**

Selain itu, penelitian ini juga menggunakan **Fisher Score Feature Selection** untuk memilih fitur-fitur yang paling relevan terhadap proses klasifikasi. Pada aplikasi B-Care, model yang digunakan adalah:

> **SVM Linear + Firefly Algorithm + Fisher Score Feature Selection**

Skenario model pada aplikasi:

- **Model:** SVM + FA + FS
- **Skenario:** Skenario 5
- **Jumlah firefly:** 5
- **Maksimum iterasi:** 15
- **Jumlah fitur terpilih:** 10 fitur
- **Validasi:** K-Fold Cross Validation

---

## 📊 Dataset

Dataset yang digunakan adalah **Breast Cancer Wisconsin Diagnostic Dataset**. Dataset ini berisi data hasil pengukuran karakteristik sel tumor payudara.

Informasi dataset:

| Keterangan | Nilai |
|---|---:|
| Nama dataset | Breast Cancer Wisconsin Diagnostic |
| Jumlah data | 569 data |
| Jumlah fitur utama | 30 fitur numerik |
| Kelas target | Benign dan Malignant |
| Format data | CSV |

Tahapan preprocessing yang dilakukan meliputi:

1. Menghapus kolom yang tidak digunakan seperti `id` dan `Unnamed: 32`.
2. Melakukan encoding label diagnosis:
   - `M` menjadi `1` atau malignant/ganas
   - `B` menjadi `-1` atau benign/jinak
3. Membagi data menjadi data training dan data testing.
4. Melakukan normalisasi menggunakan **Min-Max Normalization**.
5. Melakukan seleksi fitur menggunakan **Fisher Score**.
6. Melatih model SVM dengan parameter hasil optimasi Firefly Algorithm.

---

## ✅ Hasil Evaluasi Model

Berdasarkan hasil penelitian, model **SVM + FA + FS** memperoleh performa yang tinggi pada data uji.

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| SVM Baseline | 93.60% | 100.00% | 82.81% | 90.60% | 0.9986 |
| SVM + FS | 92.44% | 100.00% | 79.69% | 88.70% | 0.9968 |
| SVM + FA | 98.84% | 100.00% | 96.87% | 98.41% | 0.9981 |
| SVM + FA + FS | 98.84% | 100.00% | 96.87% | 98.41% | 0.9983 |

Model **SVM + FA + FS** digunakan pada aplikasi B-Care karena mampu mempertahankan performa klasifikasi yang tinggi dan menghasilkan nilai AUC terbaik dibandingkan skenario lainnya.

---

## ✨ Fitur Aplikasi

B-Care memiliki beberapa fitur utama, yaitu:

- **About**  
  Menampilkan penjelasan singkat mengenai aplikasi, tujuan, dan peringatan penggunaan.

- **Manual Input**  
  Pengguna dapat memasukkan nilai fitur secara manual berdasarkan 10 fitur hasil seleksi Fisher Score.

- **Upload CSV**  
  Pengguna dapat mengunggah file CSV yang memiliki 30 fitur Breast Cancer Wisconsin Diagnostic untuk dilakukan prediksi secara otomatis.

- **Model Information**  
  Menampilkan informasi model, skenario yang digunakan, parameter terbaik hasil Firefly Algorithm, fitur terpilih, dan hasil evaluasi model.

- **Download Hasil Prediksi**  
  Hasil prediksi dari file CSV dapat diunduh kembali dalam format CSV.

---

## 🗂️ Struktur Project

Struktur folder utama pada project ini adalah sebagai berikut:

```text
B-Care_Breast-Classification-and-Risk-Evaluation/
│
├── Buku Panduan Istalasi B-Care.pdf
├── Manual Book TA.pdf
├── jurnal dataset.pdf
│
├── Uji Coba Skenario Aplikasi/
│   ├── Percobaan_SVM_FA_FS Skenario 1.ipynb
│   ├── Percobaan_SVM_FA_FS Skenario 2.ipynb
│   ├── Percobaan_SVM_FA_FS Skenario 3.ipynb
│   ├── Percobaan_SVM_FA_FS Skenario 4.ipynb
│   ├── Percobaan_SVM_FA_FS Skenario 5.ipynb
│   ├── Percobaan_SVM_FA_FS Skenario 6.ipynb
│   └── Percobaan_SVM_FA_FS Tahap Awal.ipynb
│
└── bcare/
    ├── app.py
    ├── train_save_model.py
    ├── dataset/
    │   └── data.csv
    │
    └── bcare_streamlit_app/
        └── bcare_streamlit_app/
            ├── Images/
            │   ├── bcare_icon.png
            │   └── bcare_logo.png
            ├── model/
            │   └── bcare_svm_fa_fs.pkl
            └── requirements.txt
```

---

## ⚙️ Teknologi yang Digunakan

Project ini dibangun menggunakan:

- **Python**
- **Streamlit**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Pillow**
- **Pickle**
- **Jupyter Notebook**

---

## 🚀 Cara Menjalankan Secara Lokal

### 1. Clone Repository

```bash
git clone https://github.com/pasek2003/B-Care_Breast-Classification-and-Risk-Evaluation.git
```

Masuk ke folder project:

```bash
cd B-Care_Breast-Classification-and-Risk-Evaluation
```

---

### 2. Install Library

Install library yang dibutuhkan:

```bash
pip install -r bcare/bcare_streamlit_app/bcare_streamlit_app/requirements.txt
```

Jika terjadi error karena library gambar belum tersedia, jalankan:

```bash
pip install pillow
```

---

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run bcare/app.py
```

Setelah itu, aplikasi akan terbuka melalui browser pada alamat lokal seperti:

```text
http://localhost:8501
```

---

## 🧪 Cara Melatih Ulang Model

Jika ingin melatih ulang model dari dataset, pastikan file dataset berada pada:

```text
bcare/dataset/data.csv
```

Kemudian jalankan:

```bash
cd bcare
python train_save_model.py
```

Model hasil training akan disimpan dalam file:

```text
model/bcare_svm_fa_fs.pkl
```

---

## 🖥️ Cara Menggunakan Aplikasi

### Manual Input

1. Buka aplikasi B-Care.
2. Pilih menu **Manual Input**.
3. Masukkan nilai fitur yang diminta.
4. Klik tombol **Predict**.
5. Sistem akan menampilkan hasil klasifikasi:
   - **Benign / Jinak**
   - **Malignant / Ganas**
6. Sistem juga menampilkan **Decision Score** sebagai nilai keputusan model.

### Upload CSV

1. Pilih menu **Upload CSV**.
2. Unggah file CSV dengan format fitur Breast Cancer Wisconsin Diagnostic.
3. Sistem akan menampilkan preview data.
4. Sistem melakukan prediksi pada seluruh data.
5. Hasil prediksi dapat diunduh dalam format CSV.

---

## 📌 Catatan Penggunaan

Aplikasi ini tidak digunakan untuk diagnosis medis secara langsung. Tujuan utama aplikasi adalah:

- Implementasi hasil Tugas Akhir
- Demonstrasi model machine learning
- Media pembelajaran klasifikasi data kanker payudara
- Contoh penerapan SVM, Firefly Algorithm, dan Fisher Score Feature Selection

---

## 👨‍💻 Pengembang

**Tun Pasek Sarwiko Dipranoto**  
NIM: **2208561023**  
Program Studi Informatika  
Fakultas Matematika dan Ilmu Pengetahuan Alam  
Universitas Udayana

---

## 📄 Lisensi

Project ini dibuat untuk kebutuhan akademik dan demonstrasi penelitian. Penggunaan lebih lanjut dapat disesuaikan dengan kebutuhan pembelajaran, penelitian, dan pengembangan sistem.
