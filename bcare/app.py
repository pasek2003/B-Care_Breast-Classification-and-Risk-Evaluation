import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ============================================================
# 1. Konfigurasi path file
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

APP_DIR = BASE_DIR / "bcare_streamlit_app" / "bcare_streamlit_app"

ICON_PATH = APP_DIR / "images" / "bcare_icon.png"
LOGO_PATH = APP_DIR / "images" / "bcare_logo.png"
MODEL_PATH = APP_DIR / "model" / "bcare_svm_fa_fs.pkl"


# ============================================================
# 2. Validasi file penting
# ============================================================
if not ICON_PATH.exists():
    st.error(f"File icon tidak ditemukan di: {ICON_PATH}")
    st.stop()

if not LOGO_PATH.exists():
    st.error(f"File logo tidak ditemukan di: {LOGO_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()


# ============================================================
# 3. Konfigurasi halaman
# ============================================================
page_icon = Image.open(ICON_PATH)

st.set_page_config(
    page_title="B-Care",
    page_icon=page_icon,
    layout="wide",
)


# ============================================================
# 4. Load logo
# ============================================================
logo = Image.open(LOGO_PATH)


# ============================================================
# 5. Watermark Pembuat
# ============================================================
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 15px;
        opacity: 0.35;
        font-size: 11px;
        color: gray;
        z-index: 9999;
        text-align: right;
        line-height: 1.4;
    }
    </style>

    <div class="watermark">
        Tun Pasek Sarwiko Dipranoto<br>
        Universitas Udayana
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. Load model
# ============================================================
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        artifacts = pickle.load(file)

    return artifacts


artifacts = load_model()

w = artifacts["w"]
b = artifacts["b"]
min_vals = np.array(artifacts["min_vals"])
max_vals = np.array(artifacts["max_vals"])

selected_idx = artifacts.get("selected_idx", artifacts.get("selected_idx_fs"))
selected_idx = np.array(selected_idx)

feature_names = artifacts["feature_names"]
selected_features = artifacts.get("selected_features")

if selected_features is None:
    selected_features = [feature_names[i] for i in selected_idx]

best_params = artifacts.get("best_params", {})
metrics = artifacts.get("metrics", {})


# ============================================================
# 7. Fungsi prediksi
# ============================================================
def transform_selected_features(raw_selected_values):
    """
    Normalisasi input manual berdasarkan nilai min-max fitur terpilih.
    Input manual menggunakan nilai asli sebelum normalisasi.
    """
    raw_selected_values = np.array(raw_selected_values, dtype=float).reshape(1, -1)

    selected_min = min_vals[selected_idx]
    selected_max = max_vals[selected_idx]

    denominator = selected_max - selected_min
    denominator = np.where(denominator == 0, 1e-8, denominator)

    X_norm = (raw_selected_values - selected_min) / denominator

    return X_norm


def transform_full_features(X):
    """
    Normalisasi data dengan 30 fitur, kemudian mengambil fitur hasil Fisher Score.
    """
    X = np.array(X, dtype=float)

    denominator = max_vals - min_vals
    denominator = np.where(denominator == 0, 1e-8, denominator)

    X_norm = (X - min_vals) / denominator
    X_selected = X_norm[:, selected_idx]

    return X_selected


def svm_decision_function(X):
    return np.dot(X, w) + b


def svm_predict(X):
    scores = svm_decision_function(X)
    preds = np.sign(scores)
    preds[preds == 0] = 1

    return preds.astype(int), scores


def label_prediction(pred):
    if pred == 1:
        return "Malignant / Ganas"
    return "Benign / Jinak"


# ============================================================
# 8. Header utama
# ============================================================
col_logo, col_title = st.columns([1, 5])

with col_logo:
    st.image(logo, width=150)

with col_title:
    st.title("B-Care")
    st.caption("Breast Classification and Risk Evaluation")

st.info(
    "Aplikasi ini merupakan implementasi akademik untuk klasifikasi data Breast Cancer "
    "Wisconsin Diagnostic menggunakan model SVM Linear, Firefly Algorithm, dan Fisher Score. "
    "Hasil aplikasi bukan pengganti diagnosis medis."
)


# ============================================================
# 9. Sidebar
# ============================================================
st.sidebar.image(logo, width=130)
st.sidebar.markdown("### B-Care")

menu = st.sidebar.radio(
    "Menu",
    ["About", "Manual Input", "Upload CSV", "Model Information"],
)


# ============================================================
# 10. Menu Manual Input
# ============================================================
if menu == "Manual Input":
    st.subheader("Manual Input Data")

    st.write(
        "Masukkan nilai 10 fitur hasil seleksi Fisher Score. Nilai yang dimasukkan "
        "adalah nilai asli sebelum normalisasi."
    )

    default_values = {
        "perimeter_worst": 115.0,
        "concave points_worst": 0.10,
        "radius_worst": 16.0,
        "concave points_mean": 0.05,
        "area_worst": 850.0,
        "perimeter_mean": 90.0,
        "radius_mean": 14.0,
        "area_mean": 600.0,
        "concavity_mean": 0.08,
        "concavity_worst": 0.25,
    }

    input_values = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(selected_features):
        container = col1 if i % 2 == 0 else col2

        with container:
            value = st.number_input(
                label=feature,
                min_value=0.0,
                value=float(default_values.get(feature, 1.0)),
                step=0.01,
                format="%.6f",
            )
            input_values.append(value)

    predict_button = st.button("Predict", type="primary")

    if predict_button:
        X_input = transform_selected_features(input_values)
        pred, score = svm_predict(X_input)

        result_label = label_prediction(pred[0])

        st.divider()
        st.subheader("Hasil Klasifikasi")

        if pred[0] == 1:
            st.error(f"Hasil Prediksi: {result_label}")
        else:
            st.success(f"Hasil Prediksi: {result_label}")

        st.metric("Decision Score", f"{score[0]:.6f}")

        st.write("Data input yang digunakan:")

        input_df = pd.DataFrame(
            {
                "Feature": selected_features,
                "Input Value": input_values,
            }
        )

        st.dataframe(input_df, use_container_width=True)


# ============================================================
# 11. Menu Upload CSV
# ============================================================
elif menu == "Upload CSV":
    st.subheader("Upload CSV")

    st.write(
        "Upload file CSV yang memiliki 30 fitur Breast Cancer Wisconsin Diagnostic. "
        "Kolom `id`, `diagnosis`, dan `Unnamed: 32` boleh ada, karena akan diabaikan saat prediksi."
    )

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.write("Preview data:")
        st.dataframe(df_upload.head(), use_container_width=True)

        df_processed = df_upload.copy()

        for col in ["Unnamed: 32", "id", "diagnosis"]:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])

        missing_cols = [col for col in feature_names if col not in df_processed.columns]

        if missing_cols:
            st.error("Kolom berikut tidak ditemukan pada file CSV:")
            st.write(missing_cols)
        else:
            X_full = df_processed[feature_names].values.astype(float)
            X_selected = transform_full_features(X_full)

            preds, scores = svm_predict(X_selected)

            result_df = df_upload.copy()
            result_df["Prediction"] = [label_prediction(p) for p in preds]
            result_df["Decision Score"] = scores

            st.subheader("Hasil Prediksi")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Hasil Prediksi",
                data=csv,
                file_name="hasil_prediksi_bcare.csv",
                mime="text/csv",
            )


# ============================================================
# 12. Menu Model Information
# ============================================================
elif menu == "Model Information":
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", artifacts.get("model_name", "SVM + FA + FS"))
    col2.metric("Scenario", artifacts.get("scenario", "Skenario 5"))
    col3.metric("Selected Features", len(selected_features))

    st.write("Parameter terbaik hasil Firefly Algorithm:")
    params_df = pd.DataFrame([best_params])
    st.dataframe(params_df, use_container_width=True)

    st.write("Fitur yang digunakan oleh aplikasi:")
    feature_df = pd.DataFrame({
        "No": range(1, len(selected_features) + 1),
        "Selected Feature": selected_features,
        "Original Index": selected_idx,
    })
    
    st.dataframe(feature_df, use_container_width=True)

    st.write("Evaluasi model pada test set:")

    if metrics:
        display_metrics = {
            "Accuracy": f"{metrics.get('accuracy', 0) * 100:.2f}%",
            "Precision": f"{metrics.get('precision', 0) * 100:.2f}%",
            "Recall": f"{metrics.get('recall', 0) * 100:.2f}%",
            "F1-Score": f"{metrics.get('f1_score', 0) * 100:.2f}%",
            "AUC": f"{metrics.get('auc', 0):.4f}" if metrics.get("auc") is not None else "-",
            "TP": metrics.get("tp"),
            "TN": metrics.get("tn"),
            "FP": metrics.get("fp"),
            "FN": metrics.get("fn"),
        }

        st.dataframe(pd.DataFrame([display_metrics]), use_container_width=True)
    else:
        st.warning("Metrik evaluasi tidak tersedia pada file model.")


# ============================================================
# 13. Menu About
# ============================================================
elif menu == "About":
    st.subheader("About B-Care")

    col_about_logo, col_about_text = st.columns([1, 3])

    with col_about_logo:
        st.image(logo, width=220)

    with col_about_text:
        st.write(
            "B-Care adalah aplikasi sederhana berbasis Streamlit untuk menampilkan hasil "
            "implementasi model klasifikasi data kanker payudara. Model yang digunakan "
            "adalah Support Vector Machine Linear dengan optimasi parameter menggunakan "
            "Firefly Algorithm dan seleksi fitur menggunakan Fisher Score."
        )

        st.write(
            "Aplikasi ini menggunakan data Breast Cancer Wisconsin Diagnostic sebagai "
            "dasar pengujian model. Keluaran aplikasi berupa hasil klasifikasi ke dalam "
            "kelas benign atau malignant berdasarkan pola fitur pada data."
        )

    st.divider()

    st.subheader("Penjelasan Singkat Kanker Payudara")

    st.write(
        "Kanker payudara merupakan kondisi ketika sel pada jaringan payudara mengalami "
        "pertumbuhan tidak normal dan tidak terkendali. Sel tersebut dapat membentuk tumor "
        "dan dalam kondisi tertentu berpotensi menyebar ke jaringan tubuh lainnya."
    )

    st.info(
        "Benign menunjukkan tumor jinak, sedangkan malignant menunjukkan tumor ganas. "
        "Pada aplikasi ini, hasil klasifikasi diperoleh berdasarkan pola data pada dataset, "
        "bukan berdasarkan pemeriksaan medis secara langsung."
    )

    st.warning(
        "Aplikasi ini hanya digunakan untuk kebutuhan akademik dan demonstrasi model. "
        "Hasil klasifikasi tidak boleh digunakan sebagai dasar diagnosis medis langsung."
    )