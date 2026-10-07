"""
generate_2page_ds2_report.py
----------------------------
Page 1:
  - 22 Patients Macro Metrics Summary (Mean +- Std Dev)
  - 22 Patients Micro Metrics Summary WITH Micro Accuracy (Mean +- Std Dev)

Page 2:
  - 22 Patients Individual Macro Table
  - 22 Patients Individual Micro Table WITH Micro Accuracy
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
    PageBreak,
    HRFlowable
)
from reportlab.pdfgen import canvas

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PDF_OUTPUT_PATH = RESULTS_DIR / "DS2_Macro_and_Micro_Performance_Report.pdf"


class NumberedCanvas(canvas.Canvas):
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
        self.drawString(36, 762, "AAMI EC57 DS2 Benchmark — Macro & Micro Diagnostic Metrics (k=2 Model)")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 756, 576, 756)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 26, page_text)
        self.drawString(36, 26, "Confidential — IEEE / Biomedical Manuscript Final Evaluation Tables")
        self.line(36, 36, 576, 36)

        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT_PATH),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=38,
        bottomMargin=38
    )

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Times-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=2
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=1,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        "Header1",
        fontName="Times-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=4,
        spaceAfter=4
    )

    th_style = ParagraphStyle(
        "TableHeader",
        fontName="Times-Bold",
        fontSize=7.0,
        leading=8.5,
        textColor=colors.white,
        alignment=1
    )

    tc_style = ParagraphStyle(
        "TableCell",
        fontName="Times-Roman",
        fontSize=6.5,
        leading=7.8,
        textColor=colors.HexColor("#2D3748")
    )

    tc_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Times-Bold",
        fontSize=6.5,
        leading=7.8,
        textColor=colors.HexColor("#1A202C")
    )

    tc_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Times-Roman",
        fontSize=6.5,
        leading=7.8,
        textColor=colors.HexColor("#2D3748"),
        alignment=1
    )

    tc_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        fontName="Times-Bold",
        fontSize=6.5,
        leading=7.8,
        textColor=colors.HexColor("#1A202C"),
        alignment=1
    )

    story = []

    # =======================================================================
    # PAGE 1: 22 PATIENTS INDIVIDUAL MACRO & MICRO MEAN +- STD DEV
    # =======================================================================
    story.append(Paragraph("AAMI EC57 DS2 Benchmark: Macro & Micro Performance (k = 2 Model)", title_style))
    story.append(Paragraph("Evaluated across all 22 independent patient recordings of the AAMI EC57 DS2 partition", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1A365D"), spaceAfter=12))

    # Table 1: Macro Metrics (Mean +- Std Dev)
    story.append(Paragraph("1. DS2 22 Patients Macro Metrics Summary (Mean &plusmn; Std Dev)", h1_style))
    p1_macro_data = [
        [
            Paragraph("Evaluation Scope", th_style),
            Paragraph("Accuracy (%)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Macro Precision (%)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Macro Recall (%)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Macro F1-Score (%)<br/>(Mean &plusmn; Std)", th_style)
        ],
        [
            Paragraph("<b>All 22 DS2 Patients</b>", tc_bold),
            Paragraph("<b>90.09 &plusmn; 19.54%</b>", tc_center_bold),
            Paragraph("<b>66.59 &plusmn; 21.54%</b>", tc_center_bold),
            Paragraph("<b>66.63 &plusmn; 21.83%</b>", tc_center_bold),
            Paragraph("<b>62.44 &plusmn; 22.93%</b>", tc_center_bold)
        ],
        [
            Paragraph("<b>21 Patients (Excl. 232 Outlier)</b>", tc_style),
            Paragraph("93.29 &plusmn; 13.20%", tc_center),
            Paragraph("66.84 &plusmn; 21.99%", tc_center),
            Paragraph("67.41 &plusmn; 22.18%", tc_center),
            Paragraph("64.51 &plusmn; 21.78%", tc_center)
        ],
        [
            Paragraph("<b>Clean Cohort (20 Records, Excl. 117 &amp; 232)</b>", tc_style),
            Paragraph("95.99 &plusmn; 5.56%", tc_center),
            Paragraph("67.68 &plusmn; 22.37%", tc_center),
            Paragraph("69.79 &plusmn; 21.22%", tc_center),
            Paragraph("66.32 &plusmn; 21.28%", tc_center)
        ]
    ]
    t_p1_macro = Table(p1_macro_data, colWidths=[150, 95, 100, 95, 100])
    t_p1_macro.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EBF8FF"), colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_p1_macro)
    story.append(Spacer(1, 16))

    # Table 2: Micro Metrics WITH Micro Accuracy
    story.append(Paragraph("2. DS2 22 Patients Micro Metrics Summary (With Micro Accuracy)", h1_style))
    p1_micro_data = [
        [
            Paragraph("Micro Metric across 22 Patients", th_style),
            Paragraph("Normal (N)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("SVEB (S)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("VEB (V)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Fusion (F)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Unknown (Q)<br/>(Mean &plusmn; Std)", th_style),
            Paragraph("Overall Micro Cohort<br/>(Mean &plusmn; Std)", th_style)
        ],
        [
            Paragraph("<b>Micro Diagnostic Accuracy (%)</b>", tc_bold),
            Paragraph("<b>90.62 &plusmn; 20.28%</b>", tc_center_bold),
            Paragraph("<b>94.66 &plusmn; 16.05%</b>", tc_center_bold),
            Paragraph("<b>98.55 &plusmn; 1.85%</b>", tc_center_bold),
            Paragraph("<b>96.22 &plusmn; 12.88%</b>", tc_center_bold),
            Paragraph("<b>99.98 &plusmn; 0.04%</b>", tc_center_bold),
            Paragraph("<b>90.09 &plusmn; 19.54%</b>", tc_center_bold)
        ],
        [
            Paragraph("<b>Sensitivity / Recall (%)</b>", tc_bold),
            Paragraph("95.02 &plusmn; 13.15%", tc_center_bold),
            Paragraph("29.90 &plusmn; 32.37%", tc_center),
            Paragraph("88.47 &plusmn; 14.50%", tc_center_bold),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("---", tc_center)
        ],
        [
            Paragraph("<b>Class Precision (%)</b>", tc_bold),
            Paragraph("98.84 &plusmn; 2.46%", tc_center_bold),
            Paragraph("32.12 &plusmn; 38.55%", tc_center),
            Paragraph("80.53 &plusmn; 22.21%", tc_center_bold),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("---", tc_center)
        ],
        [
            Paragraph("<b>Class F1-Score (%)</b>", tc_bold),
            Paragraph("95.90 &plusmn; 10.54%", tc_center_bold),
            Paragraph("23.84 &plusmn; 27.55%", tc_center),
            Paragraph("82.15 &plusmn; 15.49%", tc_center_bold),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("0.00 &plusmn; 0.00%", tc_center),
            Paragraph("---", tc_center)
        ]
    ]
    t_p1_micro = Table(p1_micro_data, colWidths=[120, 70, 70, 70, 65, 65, 80])
    t_p1_micro.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#EBF8FF"), colors.white, colors.HexColor("#F8FAFC"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    story.append(t_p1_micro)

    story.append(PageBreak())

    # =======================================================================
    # PAGE 2: INDIVIDUAL MACRO & INDIVIDUAL MICRO (ALL 22 PATIENTS)
    # =======================================================================
    story.append(Paragraph("DS2 Individual Patient Breakdown: Macro & Micro Tables", title_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1A365D"), spaceAfter=6))

    # 22 patient rows for Macro
    macro_records = [
        ("100", "99.56%", "99.85%", "89.90%", "93.97%"),
        ("103", "99.90%", "49.95%", "50.00%", "49.98%"),
        ("105", "94.67%", "48.73%", "53.73%", "50.66%"),
        ("111", "99.95%", "75.00%", "99.98%", "83.32%"),
        ("113", "99.83%", "85.71%", "91.61%", "88.43%"),
        ("117*", "39.47%", "50.00%", "19.75%", "28.31%"),
        ("121", "98.23%", "44.43%", "66.09%", "49.70%"),
        ("123", "99.93%", "100.00%", "99.97%", "99.98%"),
        ("200", "97.50%", "57.65%", "63.92%", "59.72%"),
        ("202", "97.75%", "56.09%", "64.28%", "59.55%"),
        ("210", "95.77%", "47.70%", "47.42%", "47.37%"),
        ("212", "100.00%", "100.00%", "100.00%", "100.00%"),
        ("213", "87.56%", "40.86%", "48.41%", "44.28%"),
        ("214", "94.95%", "45.38%", "44.27%", "44.82%"),
        ("219", "92.57%", "40.20%", "50.98%", "43.01%"),
        ("221", "99.55%", "100.00%", "99.73%", "99.86%"),
        ("222", "76.38%", "64.96%", "79.50%", "65.61%"),
        ("228", "95.08%", "81.07%", "69.60%", "74.31%"),
        ("231", "99.87%", "66.65%", "50.00%", "55.54%"),
        ("232*", "22.81%", "61.27%", "50.29%", "18.97%"),
        ("233", "92.39%", "49.99%", "58.40%", "47.42%"),
        ("234", "98.26%", "99.42%", "68.00%", "68.94%")
    ]

    p2_macro_table_data = [
        [Paragraph("Record", th_style), Paragraph("Accuracy", th_style), Paragraph("Macro Prec.", th_style), Paragraph("Macro Rec.", th_style), Paragraph("Macro F1", th_style)]
    ]
    for r in macro_records:
        p2_macro_table_data.append([
            Paragraph(r[0], tc_bold), Paragraph(r[1], tc_center), Paragraph(r[2], tc_center), Paragraph(r[3], tc_center), Paragraph(r[4], tc_center)
        ])

    t_p2_macro = Table(p2_macro_table_data, colWidths=[38, 48, 52, 52, 48])
    t_p2_macro.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8),
    ]))

    # Individual Micro Table WITH Micro Accuracy
    micro_records = [
        ("100", "99.56%", "100.00%", "69.70%", "100.00%", "---", "---"),
        ("103", "99.90%", "100.00%", "0.00%", "---", "---", "---"),
        ("105", "94.67%", "95.32%", "---", "65.85%", "---", "0.00%"),
        ("111", "99.95%", "99.95%", "---", "100.00%", "---", "---"),
        ("113", "99.83%", "99.89%", "83.33%", "---", "---", "---"),
        ("117*", "39.47%", "39.49%", "0.00%", "---", "---", "---"),
        ("121", "98.23%", "98.28%", "0.00%", "100.00%", "---", "---"),
        ("123", "99.93%", "99.93%", "---", "100.00%", "---", "---"),
        ("200", "97.50%", "98.33%", "60.00%", "97.33%", "0.00%", "---"),
        ("202", "97.75%", "98.74%", "63.64%", "94.74%", "0.00%", "---"),
        ("210", "95.77%", "98.02%", "9.09%", "82.56%", "0.00%", "---"),
        ("212", "100.00%", "100.00%", "---", "---", "---", "---"),
        ("213", "87.56%", "100.00%", "0.00%", "93.64%", "0.00%", "---"),
        ("214", "94.95%", "97.00%", "---", "80.08%", "0.00%", "0.00%"),
        ("219", "92.57%", "92.74%", "14.29%", "96.88%", "0.00%", "---"),
        ("221", "99.55%", "99.46%", "---", "100.00%", "---", "---"),
        ("222", "76.38%", "75.75%", "83.25%", "---", "---", "---"),
        ("228", "95.08%", "99.23%", "33.33%", "76.24%", "---", "---"),
        ("231", "99.87%", "100.00%", "0.00%", "50.00%", "---", "---"),
        ("232*", "22.81%", "100.00%", "0.58%", "---", "---", "---"),
        ("233", "92.39%", "98.25%", "57.14%", "78.19%", "0.00%", "---"),
        ("234", "98.26%", "100.00%", "4.00%", "100.00%", "---", "---")
    ]

    p2_micro_table_data = [
        [Paragraph("Record", th_style), Paragraph("Micro Acc.", th_style), Paragraph("Normal (N)", th_style), Paragraph("SVEB (S)", th_style), Paragraph("VEB (V)", th_style), Paragraph("Fusion (F)", th_style), Paragraph("Unknown (Q)", th_style)]
    ]
    for r in micro_records:
        p2_micro_table_data.append([
            Paragraph(r[0], tc_bold), Paragraph(r[1], tc_center_bold), Paragraph(r[2], tc_center), Paragraph(r[3], tc_center), Paragraph(r[4], tc_center), Paragraph(r[5], tc_center), Paragraph(r[6], tc_center)
        ])

    t_p2_micro = Table(p2_micro_table_data, colWidths=[35, 46, 44, 44, 44, 42, 43])
    t_p2_micro.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 0.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0.8),
    ]))

    # Side-by-side container table
    outer_table_data = [
        [
            Paragraph("<b>Individual Macro Metrics</b>", h1_style),
            Paragraph("<b>Individual Micro Metrics (With Micro Acc.)</b>", h1_style)
        ],
        [
            t_p2_macro,
            t_p2_micro
        ]
    ]

    t_outer = Table(outer_table_data, colWidths=[240, 300])
    t_outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    story.append(t_outer)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Rebuilt Updated PDF: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    build_pdf()
