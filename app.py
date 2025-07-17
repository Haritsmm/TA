import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.db_utils import init_db, simpan_data_siswa, simpan_data_batch, ambil_semua_data, backup_db, kosongkan_database
from utils.pdf_utils import generate_pdf_report
from utils.model_utils import train_and_predict, single_predict

st.set_page_config(layout="wide")
init_db()

st.title("Prediksi Potensi Akademik Siswa (Jaringan Syaraf Tiruan)")
st.write(
    "Aplikasi profesional untuk memprediksi potensi akademik siswa (Sains, Bahasa, Sosial, Teknologi) berbasis Machine Learning. "
    "Mendukung input individu/batch, visualisasi, laporan PDF, backup data."
)

mode = st.sidebar.radio(
    "Pilih Mode",
    ("Siswa Individu", "Batch Simulasi", "Data & Visualisasi", "Database")
)

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
        potensi = st.selectbox("Potensi Asli (Label, wajib diisi untuk data latih)", ["", "Sains", "Bahasa", "Sosial", "Teknologi"], index=0, format_func=lambda x: "Pilih Potensi" if x == "" else x)
        submitted = st.form_submit_button("Simulasi & Simpan")
        if submitted:
            if (not nama or not jenis_kelamin or usia is None or 
                nilai_mtk is None or nilai_ipa is None or nilai_ips is None or 
                nilai_bindo is None or nilai_bing is None or nilai_tik is None or potensi == ""):
                st.error("Semua kolom termasuk Potensi wajib diisi dengan benar!")
            else:
                # Gunakan data berlabel untuk retrain
                df_all = ambil_semua_data()
                df_train = df_all[
                    df_all['potensi_asli'].notnull() & 
                    (df_all['potensi_asli'] != "") & 
                    (df_all['potensi_asli'] != "-")
                ] if not df_all.empty and 'potensi_asli' in df_all.columns else pd.DataFrame()
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
                        "Potensi Asli": potensi,
                        "Prediksi Potensi": hasil_pred
                    }
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
                        'potensi_asli': potensi,
                        'potensi_prediksi': hasil_pred,
                        'sumber': 'individu'
                    })
                    st.success(f"Prediksi Potensi Akademik Siswa: **{hasil_pred}** (Label Asli: {potensi})")
                    hasil_output = pd.DataFrame([st.session_state['hasil_prediksi_siswa']])
                    pdf_file = generate_pdf_report(hasil_output, f"Laporan Prediksi Siswa: {nama}")
                    st.session_state['pdf_file_siswa'] = pdf_file
    if 'hasil_prediksi_siswa' in st.session_state:
        hasil_df = pd.DataFrame([st.session_state['hasil_prediksi_siswa']])
        st.markdown("#### Preview Hasil Simulasi")
        st.dataframe(hasil_df)
        # Pie chart distribusi prediksi (termasuk prediksi terakhir)
        df_db = ambil_semua_data()
        df_all = pd.concat([
            df_db,
            hasil_df.rename(columns={"Prediksi Potensi": "potensi_prediksi"})
        ], ignore_index=True)
        st.markdown("#### Distribusi Potensi Prediksi (Pie Chart, Termasuk Simulasi Terbaru)")
        fig, ax = plt.subplots(figsize=(4, 4))
        df_all['potensi_prediksi'].value_counts().plot.pie(
            autopct='%1.0f%%', ax=ax, textprops={'fontsize': 10})
        ax.set_ylabel("")
        ax.set_title("Distribusi Potensi Prediksi", fontsize=13)
        st.pyplot(fig)
        if 'pdf_file_siswa' in st.session_state:
            with open(st.session_state['pdf_file_siswa'], "rb") as f:
                st.download_button(
                    "Download Laporan PDF",
                    f,
                    file_name="Laporan_Siswa.pdf",
                    mime="application/pdf"
                )
            import os
            os.remove(st.session_state['pdf_file_siswa'])
            del st.session_state['pdf_file_siswa']
        del st.session_state['hasil_prediksi_siswa']

elif mode == "Batch Simulasi":
    st.subheader("Upload File CSV Data Siswa")
    uploaded = st.file_uploader("Upload file CSV Data Siswa", type=["csv"])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.dataframe(df_batch.head())
        if st.button("Simulasi Batch"):
            df_all = ambil_semua_data()
            df_train = df_all[
                df_all['potensi_asli'].notnull() & 
                (df_all['potensi_asli'] != "") & 
                (df_all['potensi_asli'] != "-")
            ] if not df_all.empty and 'potensi_asli' in df_all.columns else pd.DataFrame()
            if df_train.empty:
                st.error("Data latih tidak tersedia. Harap input data latih terlebih dahulu.")
            else:
                hasil_df, acc, label_encoder, mlp, scaler = train_and_predict(df_train.copy())
                # Prediksi batch (gunakan fungsi batch predict Anda, ini contoh)
                df_batch['Prediksi Potensi'] = df_batch.apply(
                    lambda row: single_predict({
                        'Jenis_Kelamin_enc': 1 if row['Jenis Kelamin'] == "L" else 0,
                        'usia': row['Usia'],
                        'nilai_mtk': row['Nilai Matematika'],
                        'nilai_ipa': row['Nilai IPA'],
                        'nilai_ips': row['Nilai IPS'],
                        'nilai_bindo': row['Nilai Bahasa Indonesia'],
                        'nilai_bing': row['Nilai Bahasa Inggris'],
                        'nilai_tik': row['Nilai TIK'],
                        'minat_sains': row['Minat Sains'],
                        'minat_bahasa': row['Minat Bahasa'],
                        'minat_sosial': row['Minat Sosial'],
                        'minat_teknologi': row['Minat Teknologi']
                    }, mlp, scaler, label_encoder), axis=1)
                simpan_data_batch(df_batch, sumber="batch")
                st.success("Simulasi batch selesai & data tersimpan.")
                st.dataframe(df_batch)
                # Chart
                fig, ax = plt.subplots(figsize=(4, 4))
                df_batch['Prediksi Potensi'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax, textprops={'fontsize': 10})
                ax.set_ylabel("")
                ax.set_title("Distribusi Potensi Prediksi Batch", fontsize=13)
                st.pyplot(fig)
                # PDF
                pdf_file = generate_pdf_report(df_batch, "Laporan Batch Potensi Siswa")
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "Download Laporan PDF",
                        f,
                        file_name="Laporan_Batch_Potensi.pdf",
                        mime="application/pdf"
                    )
                import os
                os.remove(pdf_file)

elif mode == "Data & Visualisasi":
    st.header("Data Siswa & Visualisasi")
    df = ambil_semua_data()
    if not df.empty:
        # Rename kolom ke format rapi
        pretty_cols = {
            'nama': 'Nama', 'jenis_kelamin': 'Jenis Kelamin', 'usia': 'Usia',
            'nilai_mtk': 'Nilai Matematika', 'nilai_ipa': 'Nilai IPA', 'nilai_ips': 'Nilai IPS',
            'nilai_bindo': 'Nilai Bahasa Indonesia', 'nilai_bing': 'Nilai Bahasa Inggris', 'nilai_tik': 'Nilai TIK',
            'minat_sains': 'Minat Sains', 'minat_bahasa': 'Minat Bahasa',
            'minat_sosial': 'Minat Sosial', 'minat_teknologi': 'Minat Teknologi',
            'potensi_asli': 'Potensi Asli', 'potensi_prediksi': 'Potensi Prediksi',
            'sumber': 'Sumber', 'waktu_input': 'Waktu Input'
        }
        df_display = df.rename(columns=pretty_cols)
        st.dataframe(df_display)
        fig, ax = plt.subplots(figsize=(4, 4))
        df_display['Potensi Prediksi'].value_counts().plot.pie(
            autopct='%1.0f%%', ax=ax, textprops={'fontsize': 10})
        ax.set_ylabel("")
        ax.set_title("Distribusi Potensi Prediksi Database", fontsize=13)
        st.pyplot(fig)
        if st.button("Kosongkan Database"):
            kosongkan_database()
            st.warning("Database telah dikosongkan.")
    else:
        st.info("Database masih kosong. Silakan lakukan simulasi pada menu lain.")

elif mode == "Database":
    st.subheader("Backup Database")
    db_bytes = backup_db()
    st.download_button(
        "Download Backup Database (.db)",
        db_bytes,
        file_name="data_siswa.db",
        mime="application/octet-stream"
    )

st.markdown("---")
st.markdown("Developed as Professional Academic Project.")
