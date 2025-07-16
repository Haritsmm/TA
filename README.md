# 🎓 Prediksi Potensi Akademik Siswa SMP

Aplikasi profesional berbasis Python & Streamlit untuk **Prediksi Potensi Akademik Siswa SMP** menggunakan metode **Jaringan Syaraf Tiruan Backpropagation**.  
Menyediakan fitur prediksi otomatis, visualisasi dinamis, penyimpanan database (SQLite), backup, serta laporan PDF.  
Dikembangkan khusus untuk mendukung tugas akhir, presentasi akademik, maupun implementasi nyata di lingkungan pendidikan.

---

## ✨ Fitur Utama

- **Input Data Siswa Individu** melalui form interaktif, hasil prediksi langsung dan dapat diunduh sebagai PDF
- **Input Data Massal** (Batch): upload file CSV, proses prediksi seluruh siswa sekaligus, simpan dan download laporan (CSV/PDF)
- **Database Otomatis (SQLite)**: Semua data siswa & hasil prediksi terekam, siap backup dan migrasi
- **Visualisasi Dinamis**: Pie chart & bar chart distribusi potensi, laporan akurasi & evaluasi model (classification report)
- **Laporan PDF Otomatis**: Download laporan hasil simulasi baik individu maupun batch
- **Backup Database**: Download file database `.db` kapanpun
- **Arsitektur Modular**: Siap dikembangkan untuk migrasi ke database besar, cloud, atau deployment sekolah

---

## 🗂️ Struktur Project

potensi-akademik-siswa/
│
├─ data/
│ └─ data_siswa_smp.csv # Contoh data CSV
├─ db/
│ └─ data_siswa.db # Database SQLite (otomatis dibuat)
├─ laporan/ # (Opsional) Tempat file laporan PDF
├─ utils/
│ ├─ db_utils.py # Fungsi database
│ ├─ model_utils.py # Preprocessing & pelatihan model
│ └─ pdf_utils.py # Laporan PDF
├─ app.py # Main app Streamlit
├─ requirements.txt
├─ README.md

---

## 🚀 Cara Instalasi & Menjalankan

1. **Clone repository atau download source code**

2. **Install dependency:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Jalankan aplikasi:**
    ```bash
    streamlit run app.py
    ```

4. **Buka browser ke:**  
   

---

## ⚡ Cara Penggunaan

### 1. **Input Siswa Individu**
   - Pilih menu **"Input Siswa Individu"** di sidebar
   - Isi form data siswa, klik **Simulasi & Simpan**
   - Hasil prediksi tampil di layar dan bisa diunduh dalam bentuk PDF
   - Data & hasil otomatis tersimpan di database

### 2. **Input Data Massal (Batch Upload CSV)**
   - Pilih menu **"Batch Upload & Simulasi"**
   - Upload file `.csv` sesuai format (lihat bagian Format CSV)
   - Proses prediksi seluruh siswa langsung, hasil tampil di layar
   - Hasil bisa diunduh sebagai CSV atau PDF, seluruh data otomatis masuk database

### 3. **Visualisasi & Analisis**
   - Pilih menu **"Data & Visualisasi"**
   - Lihat semua data di database, distribusi potensi (pie/bar chart), evaluasi model, akurasi, dan classification report

### 4. **Backup Database**
   - Pilih menu **"Backup Database"**
   - Klik tombol untuk mengunduh file database `.db` (untuk backup, migrasi, atau audit data)

---

## 📝 Format Data CSV

**Wajib mencakup kolom berikut (case-insensitive):**

| Nama | Jenis Kelamin | Usia | Nilai Matematika | Nilai IPA | Nilai IPS | Nilai Bahasa Indonesia | Nilai Bahasa Inggris | Nilai TIK | Minat Sains | Minat Bahasa | Minat Sosial | Minat Teknologi | Potensi* |
|------|---------------|------|------------------|-----------|-----------|-----------------------|---------------------|-----------|-------------|--------------|--------------|-----------------|----------|
| ...  | L/P           | 12+  | 0-100            | ...       | ...       | ...                   | ...                 | ...       | 1-5         | 1-5          | 1-5          | 1-5             | (opsional) |

**Contoh:**

Nama,Jenis Kelamin,Usia,Nilai Matematika,Nilai IPA,Nilai IPS,Nilai Bahasa Indonesia,Nilai Bahasa Inggris,Nilai TIK,Minat Sains,Minat Bahasa,Minat Sosial,Minat Teknologi,Potensi
Budi Santoso,L,15,86,86,96,88,75,64,5,4,2,2,Sains
Andi Prasetyo,L,14,92,92,73,67,96,97,5,2,4,3,Sains

📊 Fitur Visualisasi
Distribusi Potensi Akademik: Pie chart & bar chart (berdasarkan seluruh data dalam database)

Akurasi & Evaluasi Model: Akurasi otomatis dihitung dari data berlabel, classification report (presisi, recall, f1-score)

Database Viewer: Lihat semua data siswa dan hasil prediksi

💡 Pengembangan Lanjutan
Migrasi ke PostgreSQL/MySQL (struktur database sudah siap)

Integrasi otentikasi user/guru

Export data/laporan rekap per semester/kelas

Analisis feature importance, tuning model, penambahan model lain

Deploy ke Streamlit Cloud, server sekolah, atau Docker

🛠️ Teknologi yang Digunakan
Python 3.9+

Streamlit (Web App Interactive)

scikit-learn (MLPClassifier/backpropagation, preprocessing)

SQLite3 (database lokal)

pandas, numpy (data processing)

matplotlib (visualisasi chart)

fpdf (generate PDF laporan)

👨‍💻 Tim Pengembang
Nama Mahasiswa: 

Program Studi: 

Dosen Pembimbing: 

Kontak: 

⚖️ Lisensi
Aplikasi ini dikembangkan untuk keperluan pendidikan, tugas akhir, dan pengembangan institusi pendidikan.
Bebas digunakan dan dimodifikasi sesuai kebutuhan.
Lisensi: MIT License
