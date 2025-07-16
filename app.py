import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Prediksi Potensi Akademik Siswa", layout="wide")

st.title("Prediksi Potensi Akademik Siswa dengan Backpropagation (MLP)")
st.write(
    "Aplikasi prediksi potensi akademik siswa berbasis jaringan syaraf tiruan (backpropagation). "
    "Fitur: input individu, batch via upload, visualisasi, dan laporan PDF."
)

# ------ FUNGSI PDF ------
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
        pdf.cell(colwidths[i], 7, c, border=1)
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

# ------ FUNGSI PREDIKSI ------
def train_and_predict(df):
    label_encoder = LabelEncoder()
    df['Potensi_enc'] = label_encoder.fit_transform(df['Potensi'])
    df['Jenis_Kelamin_enc'] = df['Jenis Kelamin'].map({'L': 1, 'P': 0})
    fitur = [
        'Jenis_Kelamin_enc', 'Usia', 'Nilai Matematika', 'Nilai IPA', 'Nilai IPS',
        'Nilai Bahasa Indonesia', 'Nilai Bahasa Inggris', 'Nilai TIK',
        'Minat Sains', 'Minat Bahasa', 'Minat Sosial', 'Minat Teknologi'
    ]
    X = df[fitur]
    y = df['Potensi_enc']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)
    acc = accuracy_score(y_test, mlp.predict(X_test))
    prediksi = mlp.predict(X_scaled)
    prediksi_label = label_encoder.inverse_transform(prediksi)
    df['Prediksi Potensi'] = prediksi_label
    return df, acc, label_encoder, mlp, scaler, fitur

def single_predict(input_data, mlp, scaler, fitur, label_encoder):
    x = np.array([input_data[fi] for fi in fitur], dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = mlp.predict(x_scaled)
    pred_label = label_encoder.inverse_transform(pred)[0]
    return pred_label

# ------ PILIH FITUR ------
st.sidebar.header("Navigasi")
mode = st.sidebar.radio(
    "Pilih Mode",
    ("Input Siswa Individu", "Batch Upload CSV & Simulasi")
)

# ------ LOAD DATA SAMPLE UNTUK TRAINING ------
@st.cache_data
def load_sample():
    df = pd.read_csv('data/data_siswa_smp.csv')
    # Interpolasi nilai numerik jika kosong
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

df_sample = load_sample()

# ------ INPUT INDIVIDU ------
if mode == "Input Siswa Individu":
    st.subheader("Input Data Siswa")
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
        # Dummy potensi untuk training (tidak digunakan pada prediksi)
        potensi = "Sains"

        submitted = st.form_submit_button("Simulasi Prediksi")
        if submitted:
            # Latih model pada sample data
            df_train, acc, label_encoder, mlp, scaler, fitur = train_and_predict(df_sample.copy())
            input_dict = {
                'Jenis_Kelamin_enc': 1 if jenis_kelamin == "L" else 0,
                'Usia': usia,
                'Nilai Matematika': nilai_mtk,
                'Nilai IPA': nilai_ipa,
                'Nilai IPS': nilai_ips,
                'Nilai Bahasa Indonesia': nilai_bindo,
                'Nilai Bahasa Inggris': nilai_bing,
                'Nilai TIK': nilai_tik,
                'Minat Sains': minat_sains,
                'Minat Bahasa': minat_bahasa,
                'Minat Sosial': minat_sosial,
                'Minat Teknologi': minat_teknologi
            }
            hasil_pred = single_predict(input_dict, mlp, scaler, fitur, label_encoder)
            st.success(f"Prediksi Potensi Akademik Siswa: **{hasil_pred}**")
            # Laporan PDF
            hasil_df = pd.DataFrame([{
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
            pdf_file = generate_pdf_report(hasil_df, f"Laporan Prediksi Siswa: {nama}")
            with open(pdf_file, "rb") as f:
                st.download_button("Download Laporan PDF", f, file_name=f"Laporan_{nama}.pdf", mime="application/pdf")
            os.remove(pdf_file)

# ------ BATCH UPLOAD & SIMULASI ------
if mode == "Batch Upload CSV & Simulasi":
    st.subheader("Upload File CSV Data Siswa")
    contoh = st.expander("Contoh format CSV", expanded=False)
    contoh.write(df_sample.head())

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
        # Interpolasi data kosong
        df.interpolate(method='linear', inplace=True)
        df.fillna(method='ffill', inplace=True)
        # Training & Prediksi
        df_pred, acc, label_encoder, mlp, scaler, fitur = train_and_predict(df.copy())
        st.success(f"Akurasi Model (uji): {acc:.2%}")

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi Potensi Akademik (Batch)")
        st.dataframe(df_pred[['Nama', 'Potensi', 'Prediksi Potensi']])

        # ---- VISUALISASI ----
        st.subheader("Visualisasi Distribusi Potensi (Data Asli vs Prediksi)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Distribusi Potensi (Label Asli):")
            fig1, ax1 = plt.subplots()
            df['Potensi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax1)
            plt.title("Label Asli")
            ax1.axis("equal")
            st.pyplot(fig1)
        with col2:
            st.write("Distribusi Potensi (Prediksi):")
            fig2, ax2 = plt.subplots()
            df_pred['Prediksi Potensi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax2)
            plt.title("Prediksi")
            ax2.axis("equal")
            st.pyplot(fig2)
        st.write("Distribusi dalam bentuk bar chart:")
        fig_bar, ax_bar = plt.subplots()
        df_pred['Prediksi Potensi'].value_counts().plot.bar(ax=ax_bar)
        plt.xlabel("Potensi")
        plt.ylabel("Jumlah Siswa")
        plt.title("Distribusi Potensi (Prediksi)")
        st.pyplot(fig_bar)

        # ---- Download hasil CSV ----
        csv_out = df_pred.to_csv(index=False).encode()
        st.download_button("Download Hasil Prediksi (CSV)", data=csv_out, file_name="hasil_prediksi_potensi.csv", mime="text/csv")
        # ---- Download PDF Laporan Batch ----
        pdf_batch = generate_pdf_report(df_pred[['Nama', 'Jenis Kelamin', 'Usia', 'Nilai Matematika',
                                                 'Nilai IPA', 'Nilai IPS', 'Nilai Bahasa Indonesia',
                                                 'Nilai Bahasa Inggris', 'Nilai TIK', 'Minat Sains',
                                                 'Minat Bahasa', 'Minat Sosial', 'Minat Teknologi',
                                                 'Potensi', 'Prediksi Potensi']])
        with open(pdf_batch, "rb") as f:
            st.download_button("Download Laporan PDF Batch", f, file_name="Laporan_Batch_Potensi.pdf", mime="application/pdf")
        os.remove(pdf_batch)
    else:
        st.info("Silakan upload file CSV atau gunakan data sample.")

# ------ VISUALISASI AKURASI (DATA SAMPLE) ------
st.markdown("---")
st.subheader("Visualisasi Akurasi Model & Distribusi Potensi (Data Sample)")
df_pred_sample, acc_sample, label_encoder, mlp, scaler, fitur = train_and_predict(df_sample.copy())
col3, col4 = st.columns(2)
with col3:
    st.metric("Akurasi Model (Sample Data)", f"{acc_sample:.2%}")
    # Show classification report as text
    y_sample = df_pred_sample['Potensi_enc']
    y_pred_sample = label_encoder.transform(df_pred_sample['Prediksi Potensi'])
    cr = classification_report(y_sample, y_pred_sample, target_names=label_encoder.classes_, output_dict=True)
    cr_df = pd.DataFrame(cr).T.iloc[:-3, :2]
    st.write("Klasifikasi Report (Sample Data):")
    st.dataframe(cr_df)

with col4:
    st.write("Distribusi Potensi (Sample Data - Prediksi):")
    fig3, ax3 = plt.subplots()
    df_pred_sample['Prediksi Potensi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax3)
    plt.title("Distribusi Prediksi (Sample Data)")
    ax3.axis("equal")
    st.pyplot(fig3)
