from fpdf import FPDF
import pandas as pd

def generate_pdf_report(df, title):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Judul
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, title, align="C", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 10, "Tanggal: " + pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'), ln=True)
    pdf.ln(2)

    # PILIH kolom utama yang ingin ditampilkan
    main_columns = [
        'Nama', 'Jenis Kelamin', 'Usia',
        'Nilai Matematika', 'Nilai IPA', 'Nilai IPS',
        'Nilai Bahasa Indonesia', 'Nilai Bahasa Inggris', 'Nilai TIK',
        'Minat Sains', 'Minat Bahasa', 'Minat Sosial', 'Minat Teknologi',
        'Potensi Asli', 'Prediksi Potensi'
    ]
    # Sesuaikan nama kolom df dengan main_columns, drop jika tak ada
    df2 = df.copy()
    df2 = df2[[col for col in main_columns if col in df2.columns]]

    # Hitung lebar kolom otomatis
    col_widths = []
    for col in df2.columns:
        lebar = max(pdf.get_string_width(str(col)) + 6, 25)
        # Cek isi tiap cell
        for isi in df2[col]:
            lebar = max(lebar, pdf.get_string_width(str(isi)) + 6)
        col_widths.append(lebar)
    total_width = sum(col_widths)
    if total_width > 285:  # A4 landscape max width
        scale = 285.0 / total_width
        col_widths = [w*scale for w in col_widths]

    # Header Tabel
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(220, 230, 240)
    for i, col in enumerate(df2.columns):
        pdf.cell(col_widths[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()

    # Isi Tabel
    pdf.set_font("Arial", "", 9)
    for idx, row in df2.iterrows():
        for i, col in enumerate(df2.columns):
            # Truncate jika terlalu panjang
            val = str(row[col])
            if len(val) > 20:
                val = val[:17] + "..."
            pdf.cell(col_widths[i], 8, val, border=1, align='C')
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 10, f"Laporan dibuat otomatis pada {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", align="R")
    output_file = "laporan_batch.pdf"
    pdf.output(output_file)
    return output_file
