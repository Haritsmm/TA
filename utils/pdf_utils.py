from fpdf import FPDF
import pandas as pd
import os

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
        # Cek logo ada atau tidak
        try:
            if os.path.exists('logo/logo-bekasi.png'):
                self.image('logo/logo-bekasi.png', 13, 8, 22)
            if os.path.exists('logo/logo-smp.png'):
                self.image('logo/logo-smp.png', 265, 8, 22)
        except Exception:
            pass

        self.set_xy(0, 10)
        self.set_font('Arial', 'B', 13)
        self.cell(0, 7, "PEMERINTAH KOTA BEKASI", align="C", ln=1)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 7, "DINAS PENDIDIKAN", align="C", ln=1)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, "SMP NEGERI 6 BEKASI", align="C", ln=1)
        self.set_font('Arial', '', 10)
        self.cell(0, 5, "Terakreditasi A / NPSN: 20222976", align="C", ln=1)
        self.cell(0, 5, "Jl. Mesjid Nurul Ihsan, Jatiwaringin, Kec. Pondokgede, Kota Bekasi.", align="C", ln=1)
        self.cell(0, 5, "Website: https://smpn6bekasi.sch.id / email: smpn6kotabekasi@gmail.com", align="C", ln=1)
        self.ln(2)
        self.set_line_width(1)
        self.line(12, self.get_y(), 287, self.get_y())
        self.ln(6)

def generate_pdf_report(df, title, kepala_sekolah="Dra.Watimah,M.M.Pd", nip="196612311995012001"):
    # Pastikan kolom sudah dimapping
    df = map_columns(df)
    pdf = PDFWithHeader(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_line_width(0.3)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 12, title.upper(), align="C", ln=True)
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
    total_width1 = sum(col_widths1)
    if total_width1 > 272:
        scale = 272.0 / total_width1
        col_widths1 = [w * scale for w in col_widths1]
    for i, col in enumerate(df1.columns):
        pdf.cell(col_widths1[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    for idx, row in df1.iterrows():
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
    total_width2 = sum(col_widths2)
    if total_width2 > 272:
        scale = 272.0 / total_width2
        col_widths2 = [w * scale for w in col_widths2]
    for i, col in enumerate(df2.columns):
        pdf.cell(col_widths2[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    for idx, row in df2.iterrows():
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

    output_file = "laporan_batch.pdf"
    pdf.output(output_file)
    return output_file
