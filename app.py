import streamlit as st
import pandas as pd
import joblib

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Prediksi Kelayakan Air Minum", layout="centered")

# Judul Aplikasi
st.title("💧 Prediksi Kelayakan Air Minum")
st.write("""
Aplikasi ini menggunakan Machine Learning dengan metode **Stacking Ensemble** untuk mengklasifikasikan apakah air aman untuk diminum atau tidak.
""")

# Load Model dan Scaler sekaligus
@st.cache_resource
def load_objects():
    try:
        model = joblib.load("model_stacking.pkl")
        scaler = joblib.load("scaler.pkl") # Memuat scaler
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {e}")
        st.warning("Pastikan file 'model_stacking.pkl' dan 'scaler.pkl' sudah ada di dalam folder proyek Anda.")
        return None, None

model, scaler = load_objects()

st.sidebar.header("Input Parameter Air")
st.sidebar.write("Masukkan nilai-nilai hasil uji kualitas air di bawah ini:")

# Membuat form input di sidebar
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

    # PERBAIKAN: Huruf besar/kecil disesuaikan persis dengan error di screenshot
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
    if model is not None and scaler is not None:
        try:
            # Mengubah input menjadi format numpy array menggunakan .values 
            # (Ini adalah trik paling ampuh agar scikit-learn tidak cerewet soal nama kolom)
            input_scaled = scaler.transform(input_df.values)
            
            # Melakukan prediksi
            prediction = model.predict(input_scaled)
            
            st.subheader("Hasil Prediksi:")
            if prediction[0] == 1:
                st.success("✅ **Air LAYAK MINUM (Potable)**")
                st.write("Berdasarkan parameter yang dimasukkan, kualitas air ini aman untuk dikonsumsi.")
            else:
                st.error("❌ **Air TIDAK LAYAK MINUM (Not Potable)**")
                st.write("Berdasarkan parameter yang dimasukkan, air ini berbahaya dan membutuhkan perawatan lebih lanjut.")
                
        except Exception as e:
            # Jika menggunakan .values masih error, kita gunakan dataframe aslinya
            try:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                st.subheader("Hasil Prediksi:")
                if prediction[0] == 1:
                    st.success("✅ **Air LAYAK MINUM (Potable)**")
                else:
                    st.error("❌ **Air TIDAK LAYAK MINUM (Not Potable)**")
            except Exception as e_inner:
                st.error(f"Terjadi kesalahan saat memproses data: {e_inner}")
