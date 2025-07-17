import sqlite3
import pandas as pd
import os

# Set path ke folder db
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

def simpan_data_siswa(data_dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO DataSiswa (
        nama, jenis_kelamin, usia, nilai_mtk, nilai_ipa, nilai_ips, nilai_bindo, nilai_bing, nilai_tik,
        minat_sains, minat_bahasa, minat_sosial, minat_teknologi,
        potensi_asli, potensi_prediksi, sumber, waktu_input
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
    """, (
        data_dict['nama'], data_dict['jenis_kelamin'], data_dict['usia'], data_dict['nilai_mtk'],
        data_dict['nilai_ipa'], data_dict['nilai_ips'], data_dict['nilai_bindo'], data_dict['nilai_bing'],
        data_dict['nilai_tik'], data_dict['minat_sains'], data_dict['minat_bahasa'], data_dict['minat_sosial'],
        data_dict['minat_teknologi'], data_dict['potensi_asli'], data_dict['potensi_prediksi'], data_dict['sumber']
    ))
    conn.commit()
    conn.close()

def simpan_data_batch(df, sumber="batch"):
    for _, row in df.iterrows():
        data_dict = {
            'nama': row.get('Nama', '-'),
            'jenis_kelamin': row.get('Jenis Kelamin', '-'),
            'usia': int(row.get('Usia', 0)),
            'nilai_mtk': float(row.get('Nilai Matematika', 0)),
            'nilai_ipa': float(row.get('Nilai IPA', 0)),
            'nilai_ips': float(row.get('Nilai IPS', 0)),
            'nilai_bindo': float(row.get('Nilai Bahasa Indonesia', 0)),
            'nilai_bing': float(row.get('Nilai Bahasa Inggris', 0)),
            'nilai_tik': float(row.get('Nilai TIK', 0)),
            'minat_sains': int(row.get('Minat Sains', 0)),
            'minat_bahasa': int(row.get('Minat Bahasa', 0)),
            'minat_sosial': int(row.get('Minat Sosial', 0)),
            'minat_teknologi': int(row.get('Minat Teknologi', 0)),
            'potensi_asli': row.get('Potensi Asli', None),
            'potensi_prediksi': row.get('Potensi Prediksi', '-'),
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
