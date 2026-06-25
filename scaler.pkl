import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

print("Mengunduh dataset Water Potability...")
# Mengunduh dataset original dari repositori publik
url = "https://raw.githubusercontent.com/sahilrahman12/Water-Potability-Prediction/main/water_potability.csv"
df = pd.read_csv(url)

# Mengisi nilai yang kosong (missing values) dengan nilai rata-rata seperti standar preprocessing
df.fillna(df.mean(), inplace=True)

# Memilih 9 kolom fitur persis seperti yang diminta oleh model Anda
fitur = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]
X = df[fitur]

print("Melatih StandardScaler...")
# Membuat dan melatih (fit) StandardScaler
scaler = StandardScaler()
scaler.fit(X)

# Menyimpan scaler ke dalam file .pkl
joblib.dump(scaler, 'scaler.pkl')

print("✅ BERHASIL! File 'scaler.pkl' telah dibuat di folder Anda.")