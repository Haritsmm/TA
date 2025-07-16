import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

FTR = [
    'Jenis_Kelamin_enc', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips',
    'nilai_bindo', 'nilai_bing', 'nilai_tik',
    'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi'
]

def preprocess_df(df):
    df['Jenis_Kelamin_enc'] = df['jenis_kelamin'].map({'L': 1, 'P': 0}).fillna(0).astype(int)
    for kolom in ['usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                  'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi']:
        df[kolom] = df[kolom].astype(float)
    return df

def train_and_predict(df):
    df = preprocess_df(df)
    label_encoder = LabelEncoder()
    df['Potensi_enc'] = label_encoder.fit_transform(df['potensi_asli'].fillna('-'))
    X = df[FTR]
    y = df['Potensi_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(y) > 3 else None
        )
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=1)
        mlp.fit(X_train, y_train)
        acc = accuracy_score(y_test, mlp.predict(X_test))
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=1)
        mlp.fit(X_scaled, y)
        acc = 1.0
    prediksi = mlp.predict(X_scaled)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    df['potensi_prediksi'] = prediksi_label
    return df, acc, label_encoder, mlp, scaler

def single_predict(input_data, mlp, scaler, label_encoder):
    x = np.array([input_data[f] for f in FTR], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = mlp.predict(x_scaled)
    pred_label = label_encoder.inverse_transform(pred)[0]
    return pred_label