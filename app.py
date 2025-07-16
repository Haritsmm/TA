import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sqlite3
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="Prediksi Potensi Akademik Siswa", layout="wide")

# ========== DATABASE SECTION ==========
DB_FILE = "data_siswa.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS DataSiswa (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama TEXT,
        jenis_kelamin TEXT,
        usia INTEGER,
        nilai_mtk REAL,
        nilai_ipa REAL,
        nilai_ips REAL,
        nilai_bindo REAL,
        nilai_bing REAL,
        nilai_tik REAL,
        minat_sains INTEGER,
        minat_bahasa INTEGER,
        minat_sosial INTEGER,
        minat_teknologi INTEGER,
        potensi_asli TEXT,
        potensi_prediksi TEXT,
        sumber TEXT,
        waktu_input TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def simpan_data_siswa(
    nama, jk, usia, mtk, ipa, ips, bindo, bing, tik,
    minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
    potensi_asli, potensi_prediksi, sumber
):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    INSERT INTO DataSiswa
    (nama, jenis_kelamin, usia, nilai_mtk, nilai_ipa, nilai_ips, nilai_bindo, nilai_bing, nilai_tik,
     minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
     potensi_asli, potensi_prediksi, sumber)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        nama, jk, usia, mtk, ipa, ips, bindo, bing, tik,
        minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
        potensi_asli, potensi_prediksi, sumber
    ))
    conn.commit()
    conn.close()

def simpan_data_batch(df, sumber="batch"):
    for _, row in df.iterrows():
        simpan_data_siswa(
            row.get("Nama", "-"), row.get("Jenis Kelamin", "-"), int(row.get("Usia", 0)),
            float(row.get("Nilai Matematika", 0)), float(row.get("Nilai IPA", 0)), float(row.get("Nilai IPS", 0)),
            float(row.get("Nilai Bahasa Indonesia", 0)), float(row.get("Nilai Bahasa Inggris", 0)), float(row.get("Nilai TIK", 0)),
            int(row.get("Minat Sains", 0)), int(row.get("Minat Bahasa", 0)), int(row.get("Minat Sosial", 0)), int(row.get("Minat Teknologi", 0)),
            row.get("Potensi", "-"), row.get("Prediksi Potensi", "-"), sumber
        )

def ambil_semua_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM DataSiswa", conn)
    conn.close()
    return df

def backup_db():
    with open(DB_FILE, "rb") as f:
        st.download_button("Backup Database (.db)", f, file_name="backup_data_siswa.db")

# ========== FUNGSI UTAMA ==========
def train_and_predict(df):
    label_encoder = LabelEncoder()
    df['Potensi_enc'] = label_encoder.fit_transform(df['potensi_asli'].fillna('-'))
    df['Jenis_Kelamin_enc'] = df['jenis_kelamin'].map({'L': 1, 'P': 0}).fillna(0).astype(int)
    fitur = [
        'Jenis_Kelamin_enc', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips',
        'nilai_bindo', 'nilai_bing', 'nilai_tik',
        'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi'
    ]
    X = df[fitur]
    y = df['Potensi_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Test split only if label more than 1 class
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
    return df, acc, label_encoder, mlp, scaler, fitur

def single_predict(input_data, mlp, scaler, fitur, label_encoder):
    x = np.array([input_data[fi] for fi in fitur], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = mlp.predict(x_scaled)
    pred_label = label_encoder.inverse_transform(pred)[0]
    return pred_label

# ========== FUNGSI PDF ==========
def generate_pdf_report(df, judul="Laporan Prediksi Potensi Akademik Siswa"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, judul, ln=1, align='C')
    pdf.set_font("Arial", size=11)
    pdf.ln(8)
    colnames = list(df.columns)
    colwidths = [35] + [30]*(len(colnames)-1)
    # Header
    for i, c in enumerate(colnames):
        pdf.cell(colwidths[i], 7, str(c), border=1)
    pdf.ln()
    # Data
    for _, row in df.iterrows():
        for i, c in enumerate(colnames):
            text = str(row[c])
            pdf.cell(colwidths[i], 7, text, border=1)
        pdf.ln()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name

# ========== INISIALISASI ==========
init_db()

st.title("Prediksi Potensi Akademik Siswa dengan Backpropagation (MLP)")
st.write(
    "Aplikasi prediksi potensi akademik siswa (Sains, Bahasa, Sosial, Teknologi) dengan penyimpanan database otomatis. "
    "Input data individu atau upload batch, model dan grafik selalu update dengan data terbaru."
)

# ========== SIDEBAR ==========
st.sidebar.header("Navigasi")
mode = st.sidebar.radio(
    "Pilih Mode",
    ("Input Siswa Individu", "Batch Upload CSV & Simulasi", "Data & Visualisasi")
)
if st.sidebar.button("Backup Database"):
    backup_db()

# ========== LOAD DATA SAMPLE UNTUK TEMPLATE (JIKA DIPERLUKAN) ==========
@st.cache_data
def load_sample():
    df = pd.read_csv('data/data_siswa_smp.csv')
    df.columns = [c.strip() for c in df.columns]
    return df

# ========== MODE 1: INPUT INDIVIDU ==========
if mode == "Input Siswa Individu":
    st.subheader("Input Data Siswa Individu")
    with st.form("form_siswa"):
        nama = st.text_input("Nama Siswa")
        jenis_kelamin = st.radio("Jenis Kelamin", ("L", "P"))
        usia = st.number_input("Usia", 10, 20, 12)
        nilai_mtk = st.number_input("Nilai Matematika", 0, 100, 80)
        nilai_ipa = st.number_input("Nilai IPA", 0, 100, 80)
        nilai_ips = st.number_input("Nilai IPS", 0, 100, 80)
        nilai_bindo = st.number_input("Nilai Bahasa Indonesia", 0, 100, 80)
        nilai_bing = st.number_input("Nilai Bahasa Inggris", 0, 100, 80)
        nilai_tik = st.number_input("Nilai TIK", 0, 100, 80)
        minat_sains = st.slider("Minat Sains (1-5)", 1, 5, 3)
        minat_bahasa = st.slider("Minat Bahasa (1-5)", 1, 5, 3)
        minat_sosial = st.slider("Minat Sosial (1-5)", 1, 5, 3)
        minat_teknologi = st.slider("Minat Teknologi (1-5)", 1, 5, 3)
        submitted = st.form_submit_button("Simulasi & Simpan")
        if submitted:
            df_all = ambil_semua_data()
            if df_all.empty:
                df_all = load_sample()
                df_all.rename(columns={
                    "Nilai Matematika": "nilai_mtk", "Nilai IPA": "nilai_ipa", "Nilai IPS": "nilai_ips",
                    "Nilai Bahasa Indonesia": "nilai_bindo", "Nilai Bahasa Inggris": "nilai_bing", "Nilai TIK": "nilai_tik",
                    "Minat Sains": "minat_sains", "Minat Bahasa": "minat_bahasa",
                    "Minat Sosial": "minat_sosial", "Minat Teknologi": "minat_teknologi",
                    "Jenis Kelamin": "jenis_kelamin", "Usia": "usia", "Potensi": "potensi_asli", "Nama": "nama"
                }, inplace=True)
            hasil_df, acc, label_encoder, mlp, scaler, fitur = train_and_predict(df_all.copy())
            input_dict = {
                'Jenis_Kelamin_enc': 1 if jenis_kelamin == "L" else 0,
                'usia': usia,
                'nilai_mtk': nilai_mtk,
                'nilai_ipa': nilai_ipa,
                'nilai_ips': nilai_ips,
                'nilai_bindo': nilai_bindo,
                'nilai_bing': nilai_bing,
                'nilai_tik': nilai_tik,
                'minat_sains': minat_sains,
                'minat_bahasa': minat_bahasa,
                'minat_sosial': minat_sosial,
                'minat_teknologi': minat_teknologi
            }
            hasil_pred = single_predict(input_dict, mlp, scaler, fitur, label_encoder)
            st.success(f"Prediksi Potensi Akademik Siswa: **{hasil_pred}**")
            # Simpan ke database
            simpan_data_siswa(
                nama, jenis_kelamin, usia, nilai_mtk, nilai_ipa, nilai_ips, nilai_bindo, nilai_bing, nilai_tik,
                minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
                None, hasil_pred, "individu"
            )
            hasil_output = pd.DataFrame([{
                "Nama": nama,
                "Jenis Kelamin": jenis_kelamin,
                "Usia": usia,
                "Nilai Matematika": nilai_mtk,
                "Nilai IPA": nilai_ipa,
                "Nilai IPS": nilai_ips,
                "Nilai Bahasa Indonesia": nilai_bindo,
                "Nilai Bahasa Inggris": nilai_bing,
                "Nilai TIK": nilai_tik,
                "Minat Sains": minat_sains,
                "Minat Bahasa": minat_bahasa,
                "Minat Sosial": minat_sosial,
                "Minat Teknologi": minat_teknologi,
                "Prediksi Potensi": hasil_pred
            }])
            # Download PDF
            pdf_file = generate_pdf_report(hasil_output, f"Laporan Prediksi Siswa: {nama}")
            with open(pdf_file, "rb") as f:
                st.download_button("Download Laporan PDF", f, file_name=f"Laporan_{nama}.pdf", mime="application/pdf")
            os.remove(pdf_file)

# ========== MODE 2: BATCH UPLOAD ==========
if mode == "Batch Upload CSV & Simulasi":
    st.subheader("Upload File CSV Data Siswa")
    contoh = st.expander("Contoh format CSV", expanded=False)
    contoh.write(load_sample().head())

    uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])
    if st.checkbox("Gunakan data sample"):
        uploaded_file = "data/data_siswa_smp.csv"

    if uploaded_file:
        if isinstance(uploaded_file, str):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.write("Data siswa di-upload:")
        st.dataframe(df)
        # Kolom normalisasi
        df.columns = [c.strip() for c in df.columns]
        # Rename agar konsisten
        df.rename(columns={
            "Nilai Matematika": "nilai_mtk", "Nilai IPA": "nilai_ipa", "Nilai IPS": "nilai_ips",
            "Nilai Bahasa Indonesia": "nilai_bindo", "Nilai Bahasa Inggris": "nilai_bing", "Nilai TIK": "nilai_tik",
            "Minat Sains": "minat_sains", "Minat Bahasa": "minat_bahasa",
            "Minat Sosial": "minat_sosial", "Minat Teknologi": "minat_teknologi",
            "Jenis Kelamin": "jenis_kelamin", "Usia": "usia", "Potensi": "potensi_asli", "Nama": "nama"
        }, inplace=True)
        # Interpolasi data kosong
        df.interpolate(method='linear', inplace=True)
        df.fillna(method='ffill', inplace=True)
        # Gabung dengan data DB lama untuk pelatihan
        df_all = ambil_semua_data()
        if df_all.empty:
            df_all = df.copy()
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
        hasil_df, acc, label_encoder, mlp, scaler, fitur = train_and_predict(df_all.copy())
        # Prediksi hanya untuk batch yang diupload (bukan seluruh data DB)
        df_pred = df.copy()
        df_pred['Jenis_Kelamin_enc'] = df_pred['jenis_kelamin'].map({'L': 1, 'P': 0}).fillna(0).astype(int)
        X_upload = df_pred[fitur]
        X_upload_scaled = scaler.transform(X_upload)
        y_upload_pred = mlp.predict(X_upload_scaled)
        prediksi_label = label_encoder.inverse_transform(y_upload_pred)
        df_pred['Prediksi Potensi'] = prediksi_label
        # Simpan batch ke database
        df_save = df_pred.copy()
        if 'potensi_asli' not in df_save.columns: df_save['potensi_asli'] = None
        simpan_data_batch(df_save, "batch")
        st.success(f"Akurasi Model (uji): {acc:.2%}")
        st.dataframe(df_pred[['nama', 'jenis_kelamin', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                              'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi', 'Potensi', 'Prediksi Potensi']])
        # Download hasil CSV
        csv_out = df_pred.to_csv(index=False).encode()
        st.download_button("Download Hasil Prediksi (CSV)", data=csv_out, file_name="hasil_prediksi_potensi.csv", mime="text/csv")
        # Download PDF laporan batch
        pdf_batch = generate_pdf_report(df_pred[['nama', 'jenis_kelamin', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                                                 'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi',
                                                 'Potensi', 'Prediksi Potensi']], "Laporan Batch Prediksi Siswa")
        with open(pdf_batch, "rb") as f:
            st.download_button("Download Laporan PDF Batch", f, file_name="Laporan_Batch_Potensi.pdf", mime="application/pdf")
        os.remove(pdf_batch)

# ========== MODE 3: DATA & VISUALISASI ==========
if mode == "Data & Visualisasi":
    st.subheader("Database & Visualisasi")
    df_db = ambil_semua_data()
    if df_db.empty:
        st.warning("Database masih kosong. Silakan input data dulu.")
    else:
        st.write(f"Jumlah seluruh data siswa dalam database: {len(df_db)}")
        st.dataframe(df_db[['nama', 'jenis_kelamin', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                            'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi',
                            'potensi_asli', 'potensi_prediksi', 'sumber', 'waktu_input']])
        # Visualisasi distribusi prediksi
        st.subheader("Distribusi Potensi Prediksi (dari seluruh data database)")
        fig1, ax1 = plt.subplots()
        df_db['potensi_prediksi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax1)
        plt.title("Distribusi Potensi Prediksi (Pie Chart)")
        ax1.axis("equal")
        st.pyplot(fig1)
        st.write("Distribusi Potensi (Bar Chart):")
        fig2, ax2 = plt.subplots()
        df_db['potensi_prediksi'].value_counts().plot.bar(ax=ax2)
        plt.xlabel("Potensi")
        plt.ylabel("Jumlah Siswa")
        plt.title("Distribusi Potensi Prediksi (Bar Chart)")
        st.pyplot(fig2)
        # Visualisasi distribusi label asli (jika tersedia)
        if df_db['potensi_asli'].notnull().any():
            st.write("Distribusi Potensi Asli (jika tersedia):")
            fig3, ax3 = plt.subplots()
            df_db['potensi_asli'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax3)
            plt.title("Distribusi Potensi Asli")
            ax3.axis("equal")
            st.pyplot(fig3)
        # Akurasi model dan classification report
        if df_db['potensi_asli'].notnull().any() and df_db['potensi_prediksi'].notnull().any():
            df_eva = df_db[df_db['potensi_asli'].notnull()]
            y_true = df_eva['potensi_asli']
            y_pred = df_eva['potensi_prediksi']
            st.subheader("Evaluasi Model (Data di Database)")
            if len(y_true.unique()) > 1:
                acc_db = accuracy_score(y_true, y_pred)
                st.metric("Akurasi (Database)", f"{acc_db:.2%}")
                cr_db = classification_report(y_true, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cr_db).T.iloc[:-3, :2]
                st.write("Classification Report:")
                st.dataframe(cr_df)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("**Database file:** `data_siswa.db` &mdash; Download backup dari sidebar jika ingin backup.")
st.markdown("Built with Love and Determine | Custom Project")
