# B-Care

B-Care adalah aplikasi sederhana berbasis Streamlit untuk klasifikasi data Breast Cancer Wisconsin Diagnostic menggunakan SVM Linear + Firefly Algorithm + Fisher Score.

## Struktur Folder

```text
bcare_streamlit_app/
├── app.py
├── train_save_model.py
├── requirements.txt
├── dataset/
│   └── data.csv
└── model/
    └── bcare_svm_fa_fs.pkl
```

## Cara Menjalankan

1. Install library:

```bash
pip install -r requirements.txt
```

2. Letakkan dataset pada:

```text
dataset/data.csv
```

3. Training dan simpan model:

```bash
python train_save_model.py
```

4. Jalankan Streamlit:

```bash
streamlit run app.py
```
