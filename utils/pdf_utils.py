from fpdf import FPDF
import pandas as pd
import os
import re
from datetime import datetime

COLUMN_MAP = {
    'nama': 'Nama',
    'jenis_kelamin': 'JK',
    'usia': 'Usia',
    'nilai_mtk': 'Matematika',
    'nilai_ipa': 'IPA',
    'nilai_ips': 'IPS',
    'nilai_bindo': 'Bahasa Indonesia',
    'nilai_bing': 'Bahasa Inggris',
    'nilai_tik': 'TIK',
    'minat_sains': 'Minat Sains',
    'minat_bahasa': 'Minat Bahasa',
    'minat_sosial': 'Minat Sosial',
    'minat_teknologi': 'Minat Teknologi',
    'potensi_asli': 'Potensi Asli',
    'potensi_prediksi': 'Potensi Prediksi'
}

def map_columns(df, colmap=COLUMN_MAP):
    return df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

class PDFWithHeader(FPDF):
    def header(self):
        # Gunakan margin konsisten untuk semua konten header
        margin = 12                # kiri/kanan
        page_w = self.w
        effective_w = page_w - 2 * margin

        # Logo kiri & kanan diposisikan SIMETRIS
        try:
            if os.path.exists('logo/logo-bekasi.png'):
                self.image('logo/logo-bekasi.png', x=margin, y=8, w=22)
            if os.path.exists('logo/logo-smp.png'):
                self.image('logo/logo-smp.png', x=page_w - margin - 22, y=8, w=22)
        except Exception:
            pass

        # Helper: cetak 1 baris tepat di tengah lebar efektif
        def center_line(txt, style='', size=12, h=7):
            self.set_font('Arial', style, size)
            self.set_x(margin)
            self.cell(effective_w, h, txt, border=0, align='C', ln=1)

        # Teks header (semua menggunakan lebar efektif yang sama)
        self.set_y(10)
        center_line("PEMERINTAH KOTA BEKASI", style='B', size=13, h=7)
        center_line("DINAS PENDIDIKAN",        style='B', size=12, h=7)
        center_line("SMP NEGERI 6 BEKASI",     style='B', size=16, h=8)
        center_line("Terakreditasi A / NPSN: 20222976", size=10, h=5)
        center_line("Jl. Mesjid Nurul Ihsan, Jatiwaringin, Kec. Pondokgede, Kota Bekasi.", size=10, h=5)
        center_line("Website: https://smpn6bekasi.sch.id / email: smpn6kotabekasi@gmail.com", size=10, h=5)

        # Garis pemisah dengan margin yang sama
        self.ln(2)
        self.set_line_width(1)
        y = self.get_y()
        self.line(margin, y, page_w - margin, y)
        self.ln(6)

def _slugify(text):
    text = re.sub(r'[^A-Za-z0-9\- _]', '', str(text))
    text = text.strip().replace(' ', '_')
    return re.sub(r'_{2,}', '_', text)

def generate_pdf_report(df, title, kepala_sekolah="Dra.Watimah,M.M.Pd", nip="196612311995012001"):
    # Pastikan kolom sudah dimapping
    df = map_columns(df)
    pdf = PDFWithHeader(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_line_width(0.3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 12, str(title).upper(), align="C", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, "Tanggal: " + pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'), ln=True)
    pdf.ln(2)

    kolom_atas = [
        'Nama', 'JK', 'Usia',
        'Matematika', 'IPA', 'IPS',
        'Bahasa Indonesia', 'Bahasa Inggris', 'TIK'
    ]
    kolom_bawah = [
        'Minat Sains', 'Minat Bahasa', 'Minat Sosial', 'Minat Teknologi',
        'Potensi Asli', 'Potensi Prediksi'
    ]
    df1 = df[[col for col in kolom_atas if col in df.columns]]
    df2 = df[[col for col in kolom_bawah if col in df.columns]]

    # Tabel 1
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(25, 118, 210)
    pdf.set_text_color(255, 255, 255)
    col_widths1 = []
    for col in df1.columns:
        lebar = max(pdf.get_string_width(str(col)) + 6, 28)
        for isi in df1[col]:
            lebar = max(lebar, pdf.get_string_width(str(isi)) + 6)
        col_widths1.append(lebar)
    total_width1 = sum(col_widths1) if col_widths1 else 0
    if total_width1 > 272 and total_width1 > 0:
        scale = 272.0 / total_width1
        col_widths1 = [w * scale for w in col_widths1]
    for i, col in enumerate(df1.columns):
        pdf.cell(col_widths1[i], 8, str(col), border=1, align='C', fill=True)
    if len(df1.columns) > 0:
        pdf.ln()
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    for _, row in df1.iterrows():
        for i, col in enumerate(df1.columns):
            val = str(row[col])
            if len(val) > 22:
                val = val[:20] + "..."
            pdf.cell(col_widths1[i], 8, val, border=1, align='C')
        pdf.ln()

    # Tabel 2
    pdf.ln(3)
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(25, 118, 210)
    pdf.set_text_color(255, 255, 255)
    col_widths2 = []
    for col in df2.columns:
        lebar = max(pdf.get_string_width(str(col)) + 6, 32)
        for isi in df2[col]:
            lebar = max(lebar, pdf.get_string_width(str(isi)) + 6)
        col_widths2.append(lebar)
    total_width2 = sum(col_widths2) if col_widths2 else 0
    if total_width2 > 272 and total_width2 > 0:
        scale = 272.0 / total_width2
        col_widths2 = [w * scale for w in col_widths2]
    for i, col in enumerate(df2.columns):
        pdf.cell(col_widths2[i], 8, str(col), border=1, align='C', fill=True)
    if len(df2.columns) > 0:
        pdf.ln()
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    for _, row in df2.iterrows():
        for i, col in enumerate(df2.columns):
            val = str(row[col])
            if len(val) > 22:
                val = val[:20] + "..."
            pdf.cell(col_widths2[i], 8, val, border=1, align='C')
        pdf.ln()

    # Footer tanda tangan
    pdf.ln(10)
    pdf.set_xy(220, pdf.get_y())
    pdf.set_font("Arial", size=11)
    pdf.cell(70, 6, f"Bekasi, {pd.Timestamp.now().strftime('%d %B %Y')}", ln=True, align="L")
    pdf.set_x(220)
    pdf.cell(70, 6, f"Kepala SMP Negeri 6 Bekasi,", ln=True, align="L")
    pdf.ln(14)
    pdf.set_x(220)
    pdf.set_font("Arial", "BU", 11)
    pdf.cell(70, 6, kepala_sekolah, ln=True, align="L")
    pdf.set_x(220)
    pdf.set_font("Arial", size=11)
    pdf.cell(70, 6, f"NIP: {nip}", ln=True, align="L")

    # Nama file unik
    slug = _slugify(title) or "laporan"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{slug}_{timestamp}.pdf"
    pdf.output(output_file)
    return output_file

