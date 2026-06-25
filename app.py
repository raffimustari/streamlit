import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Prediksi Kelayakan Air Minum", layout="centered")

# Judul Aplikasi
st.title("💧 Prediksi Kelayakan Air Minum")
st.write("""
Aplikasi ini menggunakan Machine Learning dengan metode **Stacking Ensemble (XGBoost + LightGBM + CatBoost)** untuk mengklasifikasikan apakah air aman untuk diminum atau tidak berdasarkan 9 parameter kualitas air.
""")

# Load Model (Pastikan Anda sudah menyimpan model Stacking Ensemble ke file .pkl)
# Untuk demo ini, pastikan ada file 'model_stacking.pkl' di direktori yang sama
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_stacking.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_model()

st.sidebar.header("Input Parameter Air")
st.sidebar.write("Masukkan nilai-nilai hasil uji kualitas air di bawah ini:")

# Membuat form input di sidebar untuk 9 fitur
def user_input_features():
    ph = st.sidebar.slider("1. pH Air", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.sidebar.number_input("2. Kekerasan (Hardness - mg/L)", min_value=0.0, max_value=500.0, value=196.0)
    solids = st.sidebar.number_input("3. Total Padatan Terlarut (Solids - ppm)", min_value=0.0, max_value=70000.0, value=20000.0)
    chloramines = st.sidebar.number_input("4. Kloramin (Chloramines - ppm)", min_value=0.0, max_value=15.0, value=7.0)
    sulfate = st.sidebar.number_input("5. Sulfat (Sulfate - mg/L)", min_value=0.0, max_value=500.0, value=333.0)
    conductivity = st.sidebar.number_input("6. Konduktivitas (Conductivity - μS/cm)", min_value=0.0, max_value=800.0, value=426.0)
    organic_carbon = st.sidebar.number_input("7. Karbon Organik (Organic Carbon - ppm)", min_value=0.0, max_value=30.0, value=14.0)
    trihalomethanes = st.sidebar.number_input("8. Trihalometana (Trihalomethanes - μg/L)", min_value=0.0, max_value=130.0, value=66.0)
    turbidity = st.sidebar.number_input("9. Kekeruhan (Turbidity - NTU)", min_value=0.0, max_value=10.0, value=4.0)

    # Membuat dictionary untuk DataFrame
    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Menampilkan input pengguna
st.subheader("Parameter Kualitas Air yang Dimasukkan:")
st.write(input_df)

st.write("---")

# Tombol Prediksi
if st.button("Lakukan Prediksi"):
    if model is not None:
        # Melakukan prediksi dengan model
        prediction = model.predict(input_df)
        
        # --- ATURAN TAMBAHAN (OVERRIDE) ---
        # Jika pH di bawah 6.5 (terlalu asam) atau di atas 8.5 (terlalu basa), 
        # kita paksa hasil prediksi menjadi 0 (Tidak Layak Minum)
        if input_df['ph'][0] < 6.5 or input_df['ph'][0] > 8.5:
            prediction[0] = 0
        
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.success("✅ **Air LAYAK MINUM (Potable)**")
            st.write("Berdasarkan parameter yang dimasukkan, kualitas air ini aman untuk dikonsumsi.")
        else:
            st.error("❌ **Air TIDAK LAY
