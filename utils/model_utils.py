import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# List fitur input yang digunakan model
FTR = [
    'Jenis_Kelamin_enc', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips',
    'nilai_bindo', 'nilai_bing', 'nilai_tik',
    'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi'
]

def preprocess_df(df):
    """Siapkan dataframe: ubah kolom, encode jenis_kelamin, pastikan semua numeric."""
    df = df.copy()
    # Normalisasi nama kolom jika pakai dataset luar
    rename_map = {
        "Nilai Matematika": "nilai_mtk", "Nilai IPA": "nilai_ipa", "Nilai IPS": "nilai_ips",
        "Nilai Bahasa Indonesia": "nilai_bindo", "Nilai Bahasa Inggris": "nilai_bing", "Nilai TIK": "nilai_tik",
        "Minat Sains": "minat_sains", "Minat Bahasa": "minat_bahasa",
        "Minat Sosial": "minat_sosial", "Minat Teknologi": "minat_teknologi",
        "Jenis Kelamin": "jenis_kelamin", "Usia": "usia", "Potensi": "potensi_asli", "Nama": "nama"
    }
    df.rename(columns=rename_map, inplace=True)
    # Encode jenis_kelamin: L=1, P=0
    if 'jenis_kelamin' in df.columns:
        df['Jenis_Kelamin_enc'] = df['jenis_kelamin'].map({'L': 1, 'P': 0}).fillna(0).astype(int)
    # Pastikan semua kolom numerik benar
    for col in ['usia','nilai_mtk','nilai_ipa','nilai_ips','nilai_bindo','nilai_bing','nilai_tik',
                'minat_sains','minat_bahasa','minat_sosial','minat_teknologi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def train_and_predict(df):
    """Melatih model dan menghasilkan prediksi untuk seluruh data di df."""
    df = preprocess_df(df)
    label_encoder = LabelEncoder()
    df['Potensi_enc'] = label_encoder.fit_transform(df['potensi_asli'].fillna('-'))
    X = df[FTR]
    y = df['Potensi_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kelas_count = Counter(y)
    min_per_class = min(kelas_count.values())
    # Stratified split hanya jika data cukup
    if y.nunique() > 1 and min_per_class >= 2 and len(y) >= 4:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=1)
        mlp.fit(X_train, y_train)
        acc = accuracy_score(y_test, mlp.predict(X_test))
    else:
        # fallback jika data terlalu sedikit
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=1)
        mlp.fit(X_scaled, y)
        acc = 1.0  # semua dipakai training

    prediksi = mlp.predict(X_scaled)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    df['potensi_prediksi'] = prediksi_label
    return df, acc, label_encoder, mlp, scaler

def single_predict(input_dict, mlp, scaler, label_encoder):
    # Urutkan fitur sesuai FTR
    X = np.array([[ 
        input_dict['Jenis_Kelamin_enc'],
        input_dict['usia'],
        input_dict['nilai_mtk'],
        input_dict['nilai_ipa'],
        input_dict['nilai_ips'],
        input_dict['nilai_bindo'],
        input_dict['nilai_bing'],
        input_dict['nilai_tik'],
        input_dict['minat_sains'],
        input_dict['minat_bahasa'],
        input_dict['minat_sosial'],
        input_dict['minat_teknologi'],
    ]])
    X_scaled = scaler.transform(X)
    pred = mlp.predict(X_scaled)
    label = label_encoder.inverse_transform(pred)[0]
    return label
