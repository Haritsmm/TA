import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils.db_utils import init_db, simpan_data_siswa, simpan_data_batch, ambil_semua_data, backup_db
from utils.model_utils import train_and_predict, single_predict, FTR, preprocess_df
from utils.pdf_utils import generate_pdf_report

# ========== INISIALISASI ==========
init_db()
st.set_page_config(page_title="Prediksi Potensi Akademik Siswa (Jaringan Syaraf Tiruan)", layout="wide")

st.title("Prediksi Potensi Akademik Siswa (Jaringan Syaraf Tiruan)")
st.write(
    """
    Aplikasi profesional untuk memprediksi potensi akademik siswa (**Sains, Bahasa, Sosial, Teknologi**) berbasis Machine Learning (Jaringan Syaraf Tiruan).
    """
)

st.sidebar.header("Navigasi Menu")
mode = st.sidebar.radio(
    "Pilih Mode:",
    ("Siswa Individu", "Batch Simulasi", "Data & Visualisasi", "Database")
)

# ========== LOAD DATA SAMPLE UNTUK TEMPLATE (JIKA DIPERLUKAN) ==========
@st.cache_data
def load_sample():
    df = pd.read_csv('data/data_siswa_smp.csv')
    df.columns = [c.strip() for c in df.columns]
    return df

# ========== MODE 1: SISWA INDIVIDU ==========
if mode == "Siswa Individu":
    st.subheader("Input Data Siswa Individu")
    with st.form("form_siswa"):
        nama = st.text_input("Nama Siswa")
        jenis_kelamin = st.radio("Jenis Kelamin", options=["L", "P"], index=None)
        usia = st.number_input("Usia", min_value=10, max_value=20, value=None, format="%d")
        nilai_mtk = st.number_input("Nilai Matematika", min_value=0, max_value=100, value=None, format="%d")
        nilai_ipa = st.number_input("Nilai IPA", min_value=0, max_value=100, value=None, format="%d")
        nilai_ips = st.number_input("Nilai IPS", min_value=0, max_value=100, value=None, format="%d")
        nilai_bindo = st.number_input("Nilai Bahasa Indonesia", min_value=0, max_value=100, value=None, format="%d")
        nilai_bing = st.number_input("Nilai Bahasa Inggris", min_value=0, max_value=100, value=None, format="%d")
        nilai_tik = st.number_input("Nilai TIK", min_value=0, max_value=100, value=None, format="%d")
        
        # Slider minat, default di tengah (3)
        minat_sains = st.slider("Minat Sains (1-5)", min_value=1, max_value=5, value=3)
        minat_bahasa = st.slider("Minat Bahasa (1-5)", min_value=1, max_value=5, value=3)
        minat_sosial = st.slider("Minat Sosial (1-5)", min_value=1, max_value=5, value=3)
        minat_teknologi = st.slider("Minat Teknologi (1-5)", min_value=1, max_value=5, value=3)

        submitted = st.form_submit_button("Simulasi & Simpan")

        if submitted:
            # Validasi hanya field wajib (selain slider)
            if (not nama or not jenis_kelamin or usia is None or 
                nilai_mtk is None or nilai_ipa is None or nilai_ips is None or 
                nilai_bindo is None or nilai_bing is None or nilai_tik is None):
                st.error("Semua kolom wajib diisi dengan benar!")
            else:
                # --- proses prediksi & simpan ---
                st.success("Simulasi berhasil!")


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
            hasil_df, acc, label_encoder, mlp, scaler = train_and_predict(df_all.copy())
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
            hasil_pred = single_predict(input_dict, mlp, scaler, label_encoder)
            st.success(f"Prediksi Potensi Akademik Siswa: **{hasil_pred}**")
            # Simpan ke database
            simpan_data_siswa({
                'nama': nama,
                'jenis_kelamin': jenis_kelamin,
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
                'minat_teknologi': minat_teknologi,
                'potensi_asli': None,
                'potensi_prediksi': hasil_pred,
                'sumber': 'individu'
            })
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
            pdf_file = generate_pdf_report(hasil_output, f"Laporan Prediksi Siswa: {nama}")
            with open(pdf_file, "rb") as f:
                st.download_button("Download Laporan PDF", f, file_name=f"Laporan_{nama}.pdf", mime="application/pdf")
            os.remove(pdf_file)

# ========== MODE 2: BATCH SIMULASI ==========
if mode == "Batch Simulasi":
    st.subheader("Batch Simulasi: Upload File CSV Data Siswa")
    st.info(
        "Upload file CSV berisi data siswa. "
        "Format kolom: Nama, Jenis Kelamin, Usia, Nilai Matematika, Nilai IPA, Nilai IPS, "
        "Nilai Bahasa Indonesia, Nilai Bahasa Inggris, Nilai TIK, Minat Sains, "
        "Minat Bahasa, Minat Sosial, Minat Teknologi, Potensi (opsional)"
    )
    contoh = st.expander("Contoh Format CSV", expanded=False)
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
        df.rename(columns={
            "Nilai Matematika": "nilai_mtk", "Nilai IPA": "nilai_ipa", "Nilai IPS": "nilai_ips",
            "Nilai Bahasa Indonesia": "nilai_bindo", "Nilai Bahasa Inggris": "nilai_bing", "Nilai TIK": "nilai_tik",
            "Minat Sains": "minat_sains", "Minat Bahasa": "minat_bahasa",
            "Minat Sosial": "minat_sosial", "Minat Teknologi": "minat_teknologi",
            "Jenis Kelamin": "jenis_kelamin", "Usia": "usia", "Potensi": "potensi_asli", "Nama": "nama"
        }, inplace=True)
        df = preprocess_df(df)
        df_all = ambil_semua_data()
        if df_all.empty:
            df_all = df.copy()
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)
        hasil_df, acc, label_encoder, mlp, scaler = train_and_predict(df_all.copy())
        # Prediksi hanya untuk batch yang diupload (bukan seluruh data DB)
        X_upload = df[FTR]
        X_upload_scaled = scaler.transform(X_upload)
        y_upload_pred = mlp.predict(X_upload_scaled)
        prediksi_label = label_encoder.inverse_transform(y_upload_pred)
        df['potensi_prediksi'] = prediksi_label
        simpan_data_batch(df, "batch")
        st.success(f"Akurasi Model (uji): {acc:.2%}")
        st.dataframe(df[['nama', 'jenis_kelamin', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                         'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi', 'potensi_asli', 'potensi_prediksi']])
        # Download hasil CSV & PDF
        csv_out = df.to_csv(index=False).encode()
        st.download_button("Download Hasil Prediksi (CSV)", data=csv_out, file_name="hasil_prediksi_potensi.csv", mime="text/csv")
        pdf_batch = generate_pdf_report(df[['nama', 'jenis_kelamin', 'usia', 'nilai_mtk', 'nilai_ipa', 'nilai_ips', 'nilai_bindo', 'nilai_bing', 'nilai_tik',
                                            'minat_sains', 'minat_bahasa', 'minat_sosial', 'minat_teknologi', 'potensi_asli', 'potensi_prediksi']], "Laporan Batch Prediksi Siswa")
        with open(pdf_batch, "rb") as f:
            st.download_button("Download Laporan PDF Batch", f, file_name="Laporan_Batch_Potensi.pdf", mime="application/pdf")
        os.remove(pdf_batch)

# ========== MODE 3: DATA & VISUALISASI ==========
if mode == "Data & Visualisasi":
    st.subheader("Data Siswa & Visualisasi")
    df_db = ambil_semua_data()
    if df_db.empty:
        st.warning("Database masih kosong. Silakan input data dulu.")
    else:
        st.write(f"Jumlah seluruh data siswa dalam database: {len(df_db)}")
        st.dataframe(df_db)
        st.subheader("Distribusi Potensi Prediksi (Pie & Bar Chart, seluruh data database)")
        fig1, ax1 = plt.subplots()
        df_db['potensi_prediksi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax1)
        plt.title("Distribusi Potensi Prediksi (Pie Chart)")
        ax1.axis("equal")
        st.pyplot(fig1)
        fig2, ax2 = plt.subplots()
        df_db['potensi_prediksi'].value_counts().plot.bar(ax=ax2)
        plt.xlabel("Potensi")
        plt.ylabel("Jumlah Siswa")
        plt.title("Distribusi Potensi Prediksi (Bar Chart)")
        st.pyplot(fig2)
        if df_db['potensi_asli'].notnull().any():
            st.write("Distribusi Potensi Asli (jika tersedia):")
            fig3, ax3 = plt.subplots()
            df_db['potensi_asli'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax3)
            plt.title("Distribusi Potensi Asli")
            ax3.axis("equal")
            st.pyplot(fig3)
        if df_db['potensi_asli'].notnull().any() and df_db['potensi_prediksi'].notnull().any():
            df_eva = df_db[df_db['potensi_asli'].notnull()]
            y_true = df_eva['potensi_asli']
            y_pred = df_eva['potensi_prediksi']
            st.subheader("Evaluasi Model (Data di Database)")
            if len(y_true.unique()) > 1:
                from sklearn.metrics import accuracy_score, classification_report
                acc_db = accuracy_score(y_true, y_pred)
                st.metric("Akurasi (Database)", f"{acc_db:.2%}")
                cr_db = classification_report(y_true, y_pred, output_dict=True)
                cr_df = pd.DataFrame(cr_db).T.iloc[:-3, :2]
                st.write("Classification Report:")
                st.dataframe(cr_df)

# ========== MODE 4: DATABASE ==========
if mode == "Database":
    st.subheader("Backup Database (.db)")
    dbfile = backup_db()
    st.download_button("Download data_siswa.db", dbfile, file_name="data_siswa.db")
    st.stop()

# ========== FOOTER ==========
st.markdown("---")
st.markdown("Developed as Professional Academic Project.")
