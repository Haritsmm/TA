import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import utils.db_utils as dbu
import os

from utils.model_utils import train_and_predict, single_predict, FTR, preprocess_df
from utils.db_utils import init_db, simpan_data_siswa, simpan_data_batch, ambil_semua_data, backup_db, kosongkan_database
from utils.pdf_utils import generate_pdf_report, map_columns

# ========== SETTING KUNCI ==========
KUNCI_UTAMA = "admin2025"
KUNCI_CADANGAN = "guru2025"

if "akses" not in st.session_state:
    st.session_state.akses = None

# ========== SIDEBAR KUNCI (dengan desain custom) ==========
with st.sidebar:
    st.markdown(
        """
        <div style="border:2px solid #3a3a3a; border-radius:9px; padding:14px 10px 8px 10px; margin-bottom:15px; background-color:#191C24;">
            <div style="display:flex; align-items:center; gap:10px; justify-content:center;">
                <span style="font-size:1.45em;">🔑</span>
                <span style="font-size:1.15em; font-weight:bold; letter-spacing:1px;">Key</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    password_input = st.text_input(
        "Masukkan Kunci Akses", 
        type="password", 
        placeholder="Masukkan Kunci",
        key="kunci_password"
    )
    key_clicked = st.button("Konfirmasi", use_container_width=True)
    if key_clicked:
        if password_input == KUNCI_UTAMA:
            st.session_state.akses = "semua"
            st.success("Kunci benar! Selamat Datang Admin.")
        elif password_input == KUNCI_CADANGAN:
            st.session_state.akses = "cadangan"
            st.success("Kunci benar! Selamat Datang Guru.")
        else:
            st.session_state.akses = None
            st.error("Kunci salah! Coba lagi.")

# ========== MENU AKSES OTOMATIS ==========
MENU_ALL = ["Siswa Individu", "Batch Simulasi", "Data & Visualisasi", "Database"]
MENU_LIMITED = ["Siswa Individu", "Batch Simulasi", "Data & Visualisasi"]
MENU_SINGLE = ["Siswa Individu"]

akses = st.session_state.get("akses", None)
if akses == "semua":
    menu_options = MENU_ALL
elif akses == "cadangan":
    menu_options = MENU_LIMITED
else:
    menu_options = MENU_SINGLE

# ========== PAGE SETUP & JUDUL ==========

st.set_page_config(page_title="Prediksi Potensi Akademik Siswa (Jaringan Syaraf Tiruan)", layout="wide")
init_db()

st.title("Prediksi Potensi Akademik Siswa (Jaringan Syaraf Tiruan)")
st.write(
    """
    Aplikasi untuk memprediksi potensi akademik siswa (**Sains, Bahasa, Sosial, Teknologi**) berbasis Machine Learning (Jaringan Syaraf Tiruan).
    """
)

mode = st.sidebar.radio(
    "Pilih Menu:",
    menu_options,
    key="menu"
)

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
        minat_sains = st.slider("Minat Sains (1-5)", min_value=1, max_value=5, value=3)
        minat_bahasa = st.slider("Minat Bahasa (1-5)", min_value=1, max_value=5, value=3)
        minat_sosial = st.slider("Minat Sosial (1-5)", min_value=1, max_value=5, value=3)
        minat_teknologi = st.slider("Minat Teknologi (1-5)", min_value=1, max_value=5, value=3)
        potensi = st.selectbox("Potensi Anda", 
            options=["", "Sains", "Bahasa", "Sosial", "Teknologi"], 
            index=0, format_func=lambda x: "Pilih Potensi" if x == "" else x)

        submitted = st.form_submit_button("Simulasi & Simpan")
        if submitted:
            if (not nama or not jenis_kelamin or usia is None or 
                nilai_mtk is None or nilai_ipa is None or nilai_ips is None or 
                nilai_bindo is None or nilai_bing is None or nilai_tik is None or potensi == ""):
                st.error("Semua kolom termasuk Potensi wajib diisi dengan benar!")
            else:
                df_all = ambil_semua_data()
                df_train = df_all[
                    df_all['potensi_asli'].notnull() & 
                    (df_all['potensi_asli'] != "") & 
                    (df_all['potensi_asli'] != "-")
                ] if not df_all.empty and 'potensi_asli' in df_all.columns else pd.DataFrame()
                if df_train.empty:
                    df_train = pd.read_csv('data/data_siswa_smp.csv')
                    if "Potensi" in df_train.columns:
                        df_train = df_train[
                            df_train['Potensi'].notnull() &
                            (df_train['Potensi'] != "") &
                            (df_train['Potensi'] != "-")
                        ]
                        df_train.rename(columns={"Potensi": "potensi_asli"}, inplace=True)
                if df_train.empty:
                    st.error("Data latih tidak tersedia. Harap upload data batch dengan label potensi terlebih dahulu.")
                else:
                    hasil_df, acc, label_encoder, mlp, scaler = train_and_predict(df_train.copy())
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
                    st.session_state['hasil_prediksi_siswa'] = {
                        "nama": nama,
                        "jenis_kelamin": jenis_kelamin,
                        "usia": usia,
                        "nilai_mtk": nilai_mtk,
                        "nilai_ipa": nilai_ipa,
                        "nilai_ips": nilai_ips,
                        "nilai_bindo": nilai_bindo,
                        "nilai_bing": nilai_bing,
                        "nilai_tik": nilai_tik,
                        "minat_sains": minat_sains,
                        "minat_bahasa": minat_bahasa,
                        "minat_sosial": minat_sosial,
                        "minat_teknologi": minat_teknologi,
                        "potensi_asli": potensi,
                        "potensi_prediksi": hasil_pred
                    }
                    simpan_data_siswa(st.session_state['hasil_prediksi_siswa'])
                    st.success(f"Prediksi Potensi Akademik Siswa: **{hasil_pred}** (Potensi: {potensi})")
                    hasil_output = pd.DataFrame([st.session_state['hasil_prediksi_siswa']])
                    hasil_output_out = map_columns(hasil_output)
                    pdf_file = generate_pdf_report(hasil_output_out, f"Laporan Prediksi Siswa: {nama}")
                    st.session_state['pdf_file_siswa'] = pdf_file

    if 'hasil_prediksi_siswa' in st.session_state:
        hasil_df = pd.DataFrame([st.session_state['hasil_prediksi_siswa']])
        hasil_df_out = map_columns(hasil_df)
        st.markdown("#### Preview Hasil Simulasi")
        st.dataframe(hasil_df_out)
        
        df_db = ambil_semua_data()
        df_all = pd.concat([df_db, hasil_df], ignore_index=True)
        df_all_out = map_columns(df_all)

        st.markdown("#### Distribusi Potensi Prediksi")
        col_chart, col_dummy = st.columns([1, 2])
        with col_chart:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            df_all_out['Potensi Prediksi'].value_counts().plot.pie(
                autopct='%1.0f%%', ax=ax, textprops={'fontsize': 9}
            )
            ax.set_ylabel("")
            ax.set_title("Distribusi Potensi Prediksi", fontsize=11)
            st.pyplot(fig)

    if 'pdf_file_siswa' in st.session_state:
        with open(st.session_state['pdf_file_siswa'], "rb") as f:
            st.download_button(
                "Download Laporan PDF",
                f,
                file_name="Laporan_Siswa.pdf",
                mime="application/pdf"
            )
        os.remove(st.session_state['pdf_file_siswa'])
        del st.session_state['pdf_file_siswa']
        del st.session_state['hasil_prediksi_siswa']

# ========== MODE 2: BATCH SIMULASI ==========
if mode == "Batch Simulasi":
    st.subheader("Batch Simulasi: Upload File CSV Data Siswa")
    st.info(
        "Upload file CSV berisi data siswa. Format kolom: Nama, Jenis Kelamin, Usia, Nilai Matematika, Nilai IPA, Nilai IPS, "
        "Nilai Bahasa Indonesia, Nilai Bahasa Inggris, Nilai TIK, Minat Sains, "
        "Minat Bahasa, Minat Sosial, Minat Teknologi, Potensi"
    )
    contoh = st.expander("Contoh Format CSV", expanded=False)
    contoh.write(load_sample().head())

    uploaded_file = st.file_uploader("Upload file .csv", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview Data Siswa yang Diupload:")
        st.dataframe(map_columns(df))

        if st.button("Simulasi Batch"):
            df.columns = [c.strip() for c in df.columns]
            df.rename(columns={
                "Nilai Matematika": "nilai_mtk", "Nilai IPA": "nilai_ipa", "Nilai IPS": "nilai_ips",
                "Nilai Bahasa Indonesia": "nilai_bindo", "Nilai Bahasa Inggris": "nilai_bing", "Nilai TIK": "nilai_tik",
                "Minat Sains": "minat_sains", "Minat Bahasa": "minat_bahasa",
                "Minat Sosial": "minat_sosial", "Minat Teknologi": "minat_teknologi",
                "Jenis Kelamin": "jenis_kelamin", "Usia": "usia", "Potensi": "potensi_asli", "Nama": "nama"
            }, inplace=True)
            
            if 'nama' in df.columns:
                df = df[df['nama'].astype(str).str.strip().str.lower() != "-"]
                df = df[df['nama'].astype(str).str.strip() != ""]
            df = preprocess_df(df)
            df_all = ambil_semua_data()
            if df_all.empty:
                df_all = df.copy()
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)
            hasil_df, acc, label_encoder, mlp, scaler = train_and_predict(df_all.copy())
            X_upload = df[FTR]
            X_upload_scaled = scaler.transform(X_upload)
            y_upload_pred = mlp.predict(X_upload_scaled)
            prediksi_label = label_encoder.inverse_transform(y_upload_pred)
            df['potensi_prediksi'] = prediksi_label
            simpan_data_batch(df, "batch")
            st.session_state['batch_df_result'] = df
            st.session_state['batch_acc'] = acc

    if 'batch_df_result' in st.session_state:
        df = st.session_state['batch_df_result']
        acc = st.session_state['batch_acc']
        st.success(f"Akurasi Model (uji): {acc:.2%}")

        df_view = map_columns(df)
        st.dataframe(df_view)

        csv_out = df_view.to_csv(index=False).encode()
        st.download_button(
            "Download Hasil Prediksi (CSV)",
            data=csv_out,
            file_name="hasil_prediksi_potensi.csv",
            mime="text/csv"
        )

        pdf_batch = generate_pdf_report(df_view, "Laporan Batch Prediksi Siswa")
        with open(pdf_batch, "rb") as f:
            st.download_button(
                "Download Laporan PDF Batch",
                f,
                file_name="Laporan_Batch_Potensi.pdf",
                mime="application/pdf"
            )
        os.remove(pdf_batch)
        del st.session_state['batch_df_result']
        del st.session_state['batch_acc']

# ========== MODE 3: DATA & VISUALISASI ==========
if mode == "Data & Visualisasi":
    st.subheader("Data Siswa & Visualisasi")
    df_db = ambil_semua_data()
    if df_db.empty:
        st.warning("Database masih kosong. Silakan input data dulu.")
    else:
        if st.button("Kosongkan Database", type="primary"):
            kosongkan_database()
            st.warning("Database berhasil dikosongkan! Silakan refresh halaman.")

        st.write(f"Jumlah seluruh data siswa dalam database: {len(df_db)}")
        df_db_rename = map_columns(df_db)
        st.dataframe(df_db_rename)

        st.subheader("Distribusi Potensi Prediksi")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig1, ax1 = plt.subplots(figsize=(3.2, 3.2))
            df_db_rename['Potensi Prediksi'].value_counts().plot.pie(
                labels=df_db_rename['Potensi Prediksi'].value_counts().index,
                autopct='%1.0f%%',
                textprops={'fontsize': 11},
                startangle=90,
                ax=ax1
            )
            ax1.set_title("Pie Potensi Prediksi", fontsize=14)
            ax1.axis("equal")
            fig1.tight_layout(pad=0.2)
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(3.2, 3.2))
            df_db_rename['Potensi Prediksi'].value_counts().plot.bar(ax=ax2)
            ax2.set_xlabel("Potensi")
            ax2.set_ylabel("Jumlah Siswa")
            ax2.set_title("Bar Potensi Prediksi", fontsize=14)
            fig2.tight_layout(pad=0.2)
            st.pyplot(fig2)

        with col3:
            if df_db_rename['Potensi Asli'].notnull().any():
                fig3, ax3 = plt.subplots(figsize=(3.2, 3.2))
                df_db_rename['Potensi Asli'].value_counts().plot.pie(
                    labels=df_db_rename['Potensi Asli'].value_counts().index,
                    autopct='%1.0f%%',
                    textprops={'fontsize': 11},
                    startangle=90,
                    ax=ax3
                )
                ax3.set_title("Pie Potensi Asli", fontsize=14)
                ax3.axis("equal")
                fig3.tight_layout(pad=0.2)
                st.pyplot(fig3)
            else:
                st.info("Tidak ada data Potensi Asli.")

        if df_db_rename['Potensi Asli'].notnull().any() and df_db_rename['Potensi Prediksi'].notnull().any():
            df_eva = df_db_rename[df_db_rename['Potensi Asli'].notnull()]
            y_true = df_eva['Potensi Asli']
            y_pred = df_eva['Potensi Prediksi']
            st.subheader("Evaluasi Model")
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
