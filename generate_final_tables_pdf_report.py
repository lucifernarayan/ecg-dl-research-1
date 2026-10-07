"""
generate_final_tables_pdf_report.py
------------------------------------
Compiles the concise publication-grade academic PDF report containing strictly
the exact two tables requested in the user's handwritten specification:
  Table 1: Macro Metrics (AAMI vs AAMI-kfold with Mean ± Std dev)
  Table 2: Micro / Per-Class Metrics across [N, S, V, F, Q] (AAMI vs AAMI-kfold with Mean ± Std dev)
"""

import os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_PATH = RESULTS_DIR / "ECG_Model_Validation_and_DS2_Testing_Performance_Report.pdf"


class NumberedCanvas(canvas.Canvas):
    """Adds formal running headers and page numbers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header
        self.drawString(36, 756, "Deep Learning ECG Arrhythmia Benchmark — Macro and Micro Evaluation Tables")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 750, 576, 750)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 32, page_text)
        self.drawString(36, 32, "Confidential — IEEE / Biomedical Manuscript Final Evaluation Tables")
        self.line(36, 42, 576, 42)

        self.restoreState()


def create_pdf_report():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Times-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=1,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        "Header1",
        fontName="Times-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=10,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Times-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Times-Roman",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor("#2D3748")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Times-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor("#1A202C")
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Times-Roman",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor("#2D3748"),
        alignment=1
    )

    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        fontName="Times-Bold",
        fontSize=8.0,
        leading=10.0,
        textColor=colors.HexColor("#1A202C"),
        alignment=1
    )

    story = []

    # Title & Rule
    story.append(Paragraph("ECG Arrhythmia Benchmark: Macro and Micro Performance Tables", title_style))
    story.append(Paragraph("Comparison of AAMI Independent Test (DS2) and AAMI-kfold Cross-Validation (DS1)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1A365D"), spaceAfter=12))

    # =======================================================================
    # TABLE 1: MACRO METRICS
    # =======================================================================
    story.append(Paragraph("Table 1: Macro Metrics (AAMI DS2 vs. AAMI-kfold DS1)", h1_style))
    
    table1_data = [
        [
            Paragraph("Evaluation Paradigm / Split", table_header_style),
            Paragraph("Accuracy (%)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("Macro Precision (%)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("Macro Recall (%)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("Macro F1-Score (%)<br/>(Mean &plusmn; Std)", table_header_style)
        ],
        [
            Paragraph("<b>AAMI (DS2 Test Benchmark)</b>", table_cell_bold),
            Paragraph("93.22 &plusmn; 13.88%", table_cell_center_bold),
            Paragraph("67.46 &plusmn; 22.16%", table_cell_center),
            Paragraph("68.12 &plusmn; 22.35%", table_cell_center_bold),
            Paragraph("64.12 &plusmn; 20.61%", table_cell_center_bold)
        ],
        [
            Paragraph("<b>AAMI-kfold (DS1 Validation)</b>", table_cell_bold),
            Paragraph("91.82 &plusmn; 2.09%", table_cell_center_bold),
            Paragraph("45.07 &plusmn; 12.92%", table_cell_center),
            Paragraph("44.09 &plusmn; 8.60%", table_cell_center),
            Paragraph("43.32 &plusmn; 7.72%", table_cell_center)
        ]
    ]

    t1 = Table(table1_data, colWidths=[150, 95, 100, 95, 100])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t1)
    story.append(Spacer(1, 20))

    # =======================================================================
    # TABLE 2: MICRO / PER-CLASS METRICS ACROSS [N, S, V, F, Q]
    # =======================================================================
    story.append(Paragraph("Table 2: Micro / Per-Class Metrics across 5 AAMI Labels [N, S, V, F, Q]", h1_style))
    
    table2_data = [
        [
            Paragraph("Evaluation Paradigm / Split", table_header_style),
            Paragraph("Normal (N)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("SVEB (S)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("VEB (V)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("Fusion (F)<br/>(Mean &plusmn; Std)", table_header_style),
            Paragraph("Unknown (Q)<br/>(Mean &plusmn; Std)", table_header_style)
        ],
        [
            Paragraph("<b>AAMI Sensitivity / Recall (%)</b><br/>(DS2 Test)", table_cell_bold),
            Paragraph("94.69 &plusmn; 14.34%", table_cell_center_bold),
            Paragraph("35.68 &plusmn; 33.19%", table_cell_center),
            Paragraph("89.66 &plusmn; 13.66%", table_cell_center_bold),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center)
        ],
        [
            Paragraph("<b>AAMI-kfold Sensitivity / Recall (%)</b><br/>(DS1 Validation)", table_cell_bold),
            Paragraph("97.02 &plusmn; 1.33%", table_cell_center_bold),
            Paragraph("33.51 &plusmn; 28.97%", table_cell_center),
            Paragraph("87.28 &plusmn; 8.01%", table_cell_center_bold),
            Paragraph("2.65 &plusmn; 2.28%", table_cell_center),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center)
        ],
        [
            Paragraph("<b>AAMI F1-Score (%)</b><br/>(DS2 Test)", table_cell_bold),
            Paragraph("95.90 &plusmn; 10.54%", table_cell_center_bold),
            Paragraph("23.84 &plusmn; 27.55%", table_cell_center),
            Paragraph("82.15 &plusmn; 15.49%", table_cell_center_bold),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center)
        ],
        [
            Paragraph("<b>AAMI-kfold F1-Score (%)</b><br/>(DS1 Validation)", table_cell_bold),
            Paragraph("96.70 &plusmn; 0.75%", table_cell_center_bold),
            Paragraph("37.06 &plusmn; 30.29%", table_cell_center),
            Paragraph("80.56 &plusmn; 8.29%", table_cell_center_bold),
            Paragraph("2.27 &plusmn; 1.82%", table_cell_center),
            Paragraph("0.00 &plusmn; 0.00%", table_cell_center)
        ]
    ]

    t2 = Table(table2_data, colWidths=[140, 80, 80, 80, 80, 80])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Final PDF Report successfully compiled: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    create_pdf_report()
