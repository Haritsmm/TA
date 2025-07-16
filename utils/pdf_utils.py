from fpdf import FPDF
import pandas as pd

def generate_pdf_report(df, title):
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, align="C", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "Tanggal: " + pd.Timestamp.now().strftime('%d/%m/%Y %H:%M'), ln=True)
    pdf.ln(2)
    # Table
    col_names = list(df.columns)
    col_widths = [max(pdf.get_string_width(str(col))+6, 25) for col in col_names]
    table_width = sum(col_widths)
    if table_width > 280:  # A4 landscape max width
        scale = 280.0 / table_width
        col_widths = [w*scale for w in col_widths]
    # Header
    pdf.set_font("Arial", "B", 10)
    pdf.set_fill_color(200, 220, 255)
    for i, col in enumerate(col_names):
        pdf.cell(col_widths[i], 8, str(col), border=1, align='C', fill=True)
    pdf.ln()
    # Rows
    pdf.set_font("Arial", "", 9)
    for idx, row in df.iterrows():
        for i, col in enumerate(col_names):
            pdf.cell(col_widths[i], 8, str(row[col]), border=1, align='C')
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Arial", "I", 9)
    pdf.cell(0, 10, f"Laporan dibuat otomatis pada {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}", align="R")
    output_file = "laporan_batch.pdf"
    pdf.output(output_file)
    return output_file
