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

# Load Model
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

# --- FUNGSI BARU: Logika untuk memberikan alasan ---
def cek_alasan_kualitas(df):
    alasan = []
    
    # Cek pH (Standar WHO: 6.5 - 8.5)
    if df['ph'][0] < 6.5:
        alasan.append(f"🔴 **pH ({df['ph'][0]}):** Terlalu asam. Standar normal adalah 6.5 - 8.5.")
    elif df['ph'][0] > 8.5:
        alasan.append(f"🔴 **pH ({df['ph'][0]}):** Terlalu basa. Standar normal adalah 6.5 - 8.5.")
    else:
        alasan.append(f"🟢 **pH ({df['ph'][0]}):** Berada dalam batas aman (6.5 - 8.5).")

    # Cek Turbidity / Kekeruhan (Standar WHO: ideal < 1, batas toleransi < 5)
    if df['Turbidity'][0] > 5.0:
        alasan.append(f"🔴 **Kekeruhan ({df['Turbidity'][0]} NTU):** Air terlalu keruh. Batas aman WHO adalah maksimal 5 NTU.")
    else:
        alasan.append(f"🟢 **Kekeruhan ({df['Turbidity'][0]} NTU):** Normal dan tidak keruh (≤ 5 NTU).")

    # Cek Sulfate (Standar EPA: < 250 mg/L)
    if df['Sulfate'][0] > 250:
        alasan.append(f"🔴 **Sulfat ({df['Sulfate'][0]} mg/L):** Melebihi standar aman (250 mg/L). Bisa menyebabkan gangguan pencernaan.")
    
    # Cek Chloramines (Standar EPA: < 4 mg/L)
    if df['Chloramines'][0] > 4.0:
        alasan.append(f"🔴 **Kloramin ({df['Chloramines'][0]} ppm):** Cukup tinggi dari standar batas umum (< 4 ppm).")
        
    # Cek Trihalomethanes (Standar EPA: < 80 μg/L)
    if df['Trihalomethanes'][0] > 80:
        alasan.append(f"🔴 **Trihalometana ({df['Trihalomethanes'][0]} μg/L):** Tinggi. Melebihi ambang batas aman 80 μg/L.")
        
    # Cek Solids/TDS (Standar WHO: < 1000 ppm)
    if df['Solids'][0] > 1000:
        alasan.append(f"🔴 **Total Padatan Terlarut ({df['Solids'][0]} ppm):** Sangat tinggi. Batas kelayakan minum umumnya < 1000 ppm.")

    return alasan

# Tombol Prediksi
if st.button("Lakukan Prediksi"):
    if model is not None:
        # Melakukan prediksi
        prediction = model.predict(input_df)
        
        # Mengambil alasan berdasarkan input
        alasan_list = cek_alasan_kualitas(input_df)
        
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 1:
            st.success("✅ **Air LAYAK MINUM (Potable)**")
            st.write("Model mengklasifikasikan air ini **Aman** untuk dikonsumsi. Berikut analisis ringkas parameternya:")
            
            # Tampilkan alasan (Hanya tampilkan yang hijau/aman atau peringatan ringan)
            for alasan in alasan_list:
                st.write(f"- {alasan}")
                
            st.info("💡 **Catatan:** Meskipun model memprediksi aman, tetap perhatikan standar kesehatan setempat jika terdapat peringatan (🔴) pada parameter di atas.")
                
        else:
            st.error("❌ **Air TIDAK LAYAK MINUM (Not Potable)**")
            st.write("Model mengklasifikasikan air ini **Berbahaya** untuk dikonsumsi. Kemungkinan disebabkan oleh parameter berikut:")
            
            # Tampilkan alasan
            for alasan in alasan_list:
                st.write(f"- {alasan}")
                
            st.warning("⚠️ **Tindakan Lanjut:** Air ini memerlukan purifikasi, penyaringan, atau perawatan kimiawi lebih lanjut sebelum bisa diminum.")
            
    else:
        st.warning("⚠️ Model belum dimuat. Pastikan file 'model_stacking.pkl' berada di direktori yang sama dengan aplikasi.")
