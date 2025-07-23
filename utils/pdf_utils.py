from fpdf import FPDF
import pandas as pd
import os

class PDFWithHeader(FPDF):
    def header(self):
        # Logo kiri
        self.image('logo/logo-bekasi.png', 13, 8, 22)  # x, y, width
        # Logo kanan
        self.image('logo/logo-smp.png', 265, 8, 22)    # x, y, width

        # Judul tengah
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

def generate_pdf_report(df, title, kepala_sekolah="Ahmad Yani S.Pd.M.Si", nip="197209301998031009"):
    pdf = PDFWithHeader(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_line_width(0.3)
    
    # Judul Laporan (tengah)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 12, title.upper(), align="C", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, "Tanggal: " + pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'), ln=True)
    pdf.ln(2)

    # Kolom-kolom utama (urutkan dan tampilkan yang ada saja)
    main_columns = [
        'Nama', 'Jenis Kelamin', 'Usia',
        'Nilai Matematika', 'Nilai IPA', 'Nilai IPS',
        'Nilai Bahasa Indonesia', 'Nilai Bahasa Inggris', 'Nilai TIK',
        'Minat Sains', 'Minat Bahasa', 'Minat Sosial', 'Minat Teknologi',
        'Potensi Asli', 'Potensi Prediksi'
    ]
    df2 = df.copy()
    df2 = df2[[col for col in main_columns if col in df2.columns]]

    # Lebar kolom proporsional
    col_widths = []
    for col in df2.columns:
        lebar = max(pdf.get_string_width(str(col)) + 6, 28)
        for isi in df2[col]:
            lebar = max(lebar, pdf.get_string_width(str(isi)) + 6)
        col_widths.append(lebar)
    total_width = sum(col_widths)
    if total_width > 272:  # margin L 13, R 13
        scale = 272.0 / total_width
        col_widths = [w * scale for w in col_widths]

    # Tabel Header
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(25, 118, 210)  # Biru
    pdf.set_text_color(255, 255, 255)
    for i, col in enumerate(df2.columns):
        pdf.cell(col_widths[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()

    # Isi Tabel
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    for idx, row in df2.iterrows():
        for i, col in enumerate(df2.columns):
            val = str(row[col])
            if len(val) > 22:
                val = val[:20] + "..."
            pdf.cell(col_widths[i], 8, val, border=1, align='C')
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
