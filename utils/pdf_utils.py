from fpdf import FPDF
import tempfile

def generate_pdf_report(df, judul="Laporan Prediksi Potensi Akademik Siswa"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, judul, ln=1, align='C')
    pdf.set_font("Arial", size=11)
    pdf.ln(8)
    colnames = list(df.columns)
    colwidths = [35] + [30]*(len(colnames)-1)
    # Header
    for i, c in enumerate(colnames):
        pdf.cell(colwidths[i], 7, str(c), border=1)
    pdf.ln()
    # Data
    for _, row in df.iterrows():
        for i, c in enumerate(colnames):
            text = str(row[c])
            pdf.cell(colwidths[i], 7, text, border=1)
        pdf.ln()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        return tmpfile.name