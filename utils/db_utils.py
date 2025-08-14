import os
import sqlite3
import pandas as pd

# Path database relatif dari root project
DB_FOLDER = 'db'
DB_PATH = os.path.join(DB_FOLDER, 'data_siswa.db')

# Pastikan folder db ada
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS DataSiswa (
        id INTEGER PRIMARY KEY,
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
        waktu_input TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def _pick(row, *keys, default=None):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default

def simpan_data_siswa(data_dict):
    sumber = data_dict.get('sumber', 'individu')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO DataSiswa (
        nama, jenis_kelamin, usia, nilai_mtk, nilai_ipa, nilai_ips, nilai_bindo, nilai_bing, nilai_tik,
        minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
        potensi_asli, potensi_prediksi, sumber, waktu_input
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
    """, (
        data_dict.get('nama', '-'),
        data_dict.get('jenis_kelamin', '-'),
        int(data_dict.get('usia', 0) or 0),
        float(data_dict.get('nilai_mtk', 0) or 0.0),
        float(data_dict.get('nilai_ipa', 0) or 0.0),
        float(data_dict.get('nilai_ips', 0) or 0.0),
        float(data_dict.get('nilai_bindo', 0) or 0.0),
        float(data_dict.get('nilai_bing', 0) or 0.0),
        float(data_dict.get('nilai_tik', 0) or 0.0),
        int(data_dict.get('minat_sains', 0) or 0),
        int(data_dict.get('minat_bahasa', 0) or 0),
        int(data_dict.get('minat_sosial', 0) or 0),
        int(data_dict.get('minat_teknologi', 0) or 0),
        data_dict.get('potensi_asli'),
        data_dict.get('potensi_prediksi', '-'),
        sumber
    ))
    conn.commit()
    conn.close()

def simpan_data_batch(df, sumber="batch"):
    # Terima df dengan snake_case ATAU Title Case
    for _, row in df.iterrows():
        r = row if isinstance(row, pd.Series) else pd.Series(row)
        data_dict = {
            'nama': _pick(r, 'nama', 'Nama', default='-'),
            'jenis_kelamin': _pick(r, 'jenis_kelamin', 'Jenis Kelamin', default='-'),
            'usia': _pick(r, 'usia', 'Usia', default=0),
            'nilai_mtk': _pick(r, 'nilai_mtk', 'Nilai Matematika', default=0),
            'nilai_ipa': _pick(r, 'nilai_ipa', 'Nilai IPA', default=0),
            'nilai_ips': _pick(r, 'nilai_ips', 'Nilai IPS', default=0),
            'nilai_bindo': _pick(r, 'nilai_bindo', 'Nilai Bahasa Indonesia', default=0),
            'nilai_bing': _pick(r, 'nilai_bing', 'Nilai Bahasa Inggris', default=0),
            'nilai_tik': _pick(r, 'nilai_tik', 'Nilai TIK', default=0),
            'minat_sains': _pick(r, 'minat_sains', 'Minat Sains', default=0),
            'minat_bahasa': _pick(r, 'minat_bahasa', 'Minat Bahasa', default=0),
            'minat_sosial': _pick(r, 'minat_sosial', 'Minat Sosial', default=0),
            'minat_teknologi': _pick(r, 'minat_teknologi', 'Minat Teknologi', default=0),
            'potensi_asli': _pick(r, 'potensi_asli', 'Potensi', 'Potensi Asli', default=None),
            'potensi_prediksi': _pick(r, 'potensi_prediksi', 'Potensi Prediksi', default='-'),
            'sumber': sumber
        }
        simpan_data_siswa(data_dict)

def ambil_semua_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM DataSiswa", conn)
    conn.close()
    return df

def backup_db():
    with open(DB_PATH, "rb") as f:
        return f.read()

def kosongkan_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM DataSiswa")
    conn.commit()
    conn.close()
