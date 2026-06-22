import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Water Potability Checker",
    page_icon="💧",
    layout="wide"
)

# ==========================================
# 2. FUNGSI UNTUK LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    with open('model_air.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# ==========================================
# 3. SIDEBAR (NAVIGASI)
# ==========================================
st.sidebar.title("💧 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Prediksi Kelayakan Air", "📊 Eksplorasi Data (EDA)"])

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Machine Learning untuk memprediksi kelayakan air minum berdasarkan parameter kimia dan fisik air.")

# ==========================================
# 4. HALAMAN 1: PREDIKSI KELAYAKAN AIR
# ==========================================
if menu == "🏠 Prediksi Kelayakan Air":
    st.title("Prediksi Kelayakan Air Minum 🚰")
    st.write("Masukkan nilai parameter sampel air pada form di bawah ini untuk mengetahui apakah air tersebut aman untuk diminum.")

    # Membuat Form Input dalam 3 Kolom agar rapi
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("Tingkat pH (0-14)", min_value=0.0, max_value=14.0, value=7.0)
        hardness = st.number_input("Hardness (Kekerasan) mg/L", min_value=0.0, value=150.0)
        solids = st.number_input("Solids (TDS) ppm", min_value=0.0, value=20000.0)

    with col2:
        chloramines = st.number_input("Kloramin (ppm)", min_value=0.0, value=7.0)
        sulfate = st.number_input("Sulfat (mg/L)", min_value=0.0, value=300.0)
        conductivity = st.number_input("Konduktivitas (μS/cm)", min_value=0.0, value=400.0)

    with col3:
        organic_carbon = st.number_input("Karbon Organik (ppm)", min_value=0.0, value=10.0)
        trihalomethanes = st.number_input("Trihalometana (μg/L)", min_value=0.0, value=60.0)
        turbidity = st.number_input("Kekeruhan (NTU)", min_value=0.0, value=4.0)

    # Tombol Prediksi
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Prediksi Kelayakan", type="primary", use_container_width=True):
        
        # 1. Mengumpulkan data dari form
        input_data = pd.DataFrame({
            'ph': [ph], 'Hardness': [hardness], 'Solids': [solids],
            'Chloramines': [chloramines], 'Sulfate': [sulfate], 'Conductivity': [conductivity],
            'Organic_carbon': [organic_carbon], 'Trihalomethanes': [trihalomethanes], 'Turbidity': [turbidity]
        })
        
        # 2. PROSES PREDIKSI MENGGUNAKAN MODEL ML ASLI
        with st.spinner('Memproses data dengan Machine Learning...'):
            prediction = model.predict(input_data)
            result = prediction[0]

        # 3. Menampilkan Hasil
        st.markdown("---")
        if result == 1:
            st.success("✅ **HASIL: AIR AMAN DIMINUM (Potable)**")
            st.write("Berdasarkan parameter yang dimasukkan, kualitas air memenuhi standar kelayakan minum.")
        else:
            st.error("⚠️ **HASIL: AIR TIDAK AMAN DIMINUM (Not Potable)**")
            st.write("Perhatian! Air ini terindikasi memiliki parameter yang berada di luar batas aman konsumsi.")

# ==========================================
# 5. HALAMAN 2: EKSPLORASI DATA (EDA)
# ==========================================
elif menu == "📊 Eksplorasi Data (EDA)":
    st.title("Eksplorasi Data Kualitas Air 📈")
    st.write("Halaman ini menampilkan visualisasi interaktif dari dataset air minum.")
    
    # Memuat Dataset Asli (Ganti 'water_potability.csv' dengan nama file kamu)
    try:
        # df = pd.read_csv('water_potability.csv') # Uncomment ini jika file csv-nya ada
        
        # --- DATA DUMMY (Hapus bagian ini jika menggunakan dataset asli) ---
        np.random.seed(42)
        df = pd.DataFrame({
            'ph': np.random.normal(7, 1.5, 500),
            'Turbidity': np.random.normal(4, 1, 500),
            'Potability': np.random.choice(['Aman', 'Tidak Aman'], 500)
        })
        # ------------------------------------------------------------------

        # Menampilkan tabel data
        st.subheader("Data Sampel")
        st.dataframe(df.head(10))

        # Visualisasi 1: Distribusi pH menggunakan Plotly
        st.subheader("1. Distribusi Tingkat pH Air")
        fig_ph = px.histogram(df, x="ph", color="Potability", barmode="overlay", 
                              title="Distribusi pH berdasarkan Kelayakan",
                              color_discrete_map={'Aman': '#00CC96', 'Tidak Aman': '#EF553B'})
        st.plotly_chart(fig_ph, use_container_width=True)

        # Visualisasi 2: Scatter Plot (Hubungan pH dan Kekeruhan)
        st.subheader("2. Hubungan pH dan Kekeruhan (Turbidity)")
        fig_scatter = px.scatter(df, x="ph", y="Turbidity", color="Potability",
                                 title="pH vs Turbidity", opacity=0.7,
                                 color_discrete_map={'Aman': '#00CC96', 'Tidak Aman': '#EF553B'})
        st.plotly_chart(fig_scatter, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ File dataset tidak ditemukan. Pastikan file CSV kamu ada di folder yang sama dengan app.py.")
