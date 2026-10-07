"""
generate_ds2_macro_micro_pdf.py
--------------------------------
Generates a clean publication-grade PDF containing strictly:
  Table 1: DS2 Patient-Wise Macro Metrics (All 22 Patients + Mean +- Std Dev)
  Table 2: DS2 Patient-Wise Micro / Per-Class Sensitivity (All 22 Patients + Mean +- Std Dev)
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
    """Running header and footer with page count."""
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
        self.drawString(36, 756, "AAMI EC57 DS2 Benchmark — Patient-Wise Macro & Micro Diagnostic Metrics")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 750, 576, 750)

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 32, page_text)
        self.drawString(36, 32, "Confidential — Research Benchmark Report")
        self.line(36, 42, 576, 42)

        self.restoreState()


def create_pdf():
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
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Roman",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#2B6CB0"),
        alignment=1,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        "Header1",
        fontName="Times-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=4,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="Times-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        fontName="Times-Roman",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#2D3748")
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        fontName="Times-Bold",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#1A202C")
    )

    table_cell_center = ParagraphStyle(
        "TableCellCenter",
        fontName="Times-Roman",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#2D3748"),
        alignment=1
    )

    table_cell_center_bold = ParagraphStyle(
        "TableCellCenterBold",
        fontName="Times-Bold",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#1A202C"),
        alignment=1
    )

    story = []

    # =======================================================================
    # PAGE 1: TABLE 1 - DS2 PATIENT-WISE MACRO METRICS
    # =======================================================================
    story.append(Paragraph("DS2 Independent Test Benchmark: Patient-Wise Macro Metrics", title_style))
    story.append(Paragraph("Evaluated across all 22 independent patient recordings of the AAMI EC57 DS2 partition", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1A365D"), spaceAfter=8))

    story.append(Paragraph("Table 1: DS2 Patient-Wise Macro Metrics (All 22 Patients)", h1_style))

    t1_rows = [
        [Paragraph("Patient Record", table_header_style),
         Paragraph("Accuracy (%)", table_header_style),
         Paragraph("Macro Precision (%)", table_header_style),
         Paragraph("Macro Recall (%)", table_header_style),
         Paragraph("Macro F1-Score (%)", table_header_style)],

        [Paragraph("100", table_cell_bold), Paragraph("99.56%", table_cell_center), Paragraph("99.85%", table_cell_center), Paragraph("89.90%", table_cell_center), Paragraph("93.97%", table_cell_center)],
        [Paragraph("103", table_cell_bold), Paragraph("99.86%", table_cell_center), Paragraph("49.95%", table_cell_center), Paragraph("49.98%", table_cell_center), Paragraph("49.96%", table_cell_center)],
        [Paragraph("105", table_cell_bold), Paragraph("94.82%", table_cell_center), Paragraph("49.90%", table_cell_center), Paragraph("56.18%", table_cell_center), Paragraph("52.30%", table_cell_center)],
        [Paragraph("111", table_cell_bold), Paragraph("99.95%", table_cell_center), Paragraph("75.00%", table_cell_center), Paragraph("99.98%", table_cell_center), Paragraph("83.32%", table_cell_center)],
        [Paragraph("113", table_cell_bold), Paragraph("99.78%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("66.67%", table_cell_center), Paragraph("75.00%", table_cell_center)],
        [Paragraph("117", table_cell_bold), Paragraph("33.59%", table_cell_center), Paragraph("50.00%", table_cell_center), Paragraph("16.81%", table_cell_center), Paragraph("25.16%", table_cell_center)],
        [Paragraph("121", table_cell_bold), Paragraph("98.33%", table_cell_center), Paragraph("45.56%", table_cell_center), Paragraph("99.44%", table_cell_center), Paragraph("51.87%", table_cell_center)],
        [Paragraph("123", table_cell_bold), Paragraph("99.93%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("99.97%", table_cell_center), Paragraph("99.98%", table_cell_center)],
        [Paragraph("200", table_cell_bold), Paragraph("97.88%", table_cell_center), Paragraph("58.84%", table_cell_center), Paragraph("64.17%", table_cell_center), Paragraph("60.80%", table_cell_center)],
        [Paragraph("202", table_cell_bold), Paragraph("97.84%", table_cell_center), Paragraph("56.00%", table_cell_center), Paragraph("64.30%", table_cell_center), Paragraph("59.41%", table_cell_center)],
        [Paragraph("210", table_cell_bold), Paragraph("96.18%", table_cell_center), Paragraph("47.77%", table_cell_center), Paragraph("48.83%", table_cell_center), Paragraph("48.17%", table_cell_center)],
        [Paragraph("212", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("100.00%", table_cell_center)],
        [Paragraph("213", table_cell_bold), Paragraph("87.65%", table_cell_center), Paragraph("40.89%", table_cell_center), Paragraph("48.75%", table_cell_center), Paragraph("44.43%", table_cell_center)],
        [Paragraph("214", table_cell_bold), Paragraph("94.47%", table_cell_center), Paragraph("44.27%", table_cell_center), Paragraph("44.30%", table_cell_center), Paragraph("44.27%", table_cell_center)],
        [Paragraph("219", table_cell_bold), Paragraph("92.61%", table_cell_center), Paragraph("39.26%", table_cell_center), Paragraph("50.99%", table_cell_center), Paragraph("42.30%", table_cell_center)],
        [Paragraph("221", table_cell_bold), Paragraph("99.63%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("99.78%", table_cell_center), Paragraph("99.89%", table_cell_center)],
        [Paragraph("222", table_cell_bold), Paragraph("79.36%", table_cell_center), Paragraph("66.04%", table_cell_center), Paragraph("76.78%", table_cell_center), Paragraph("67.21%", table_cell_center)],
        [Paragraph("228", table_cell_bold), Paragraph("95.27%", table_cell_center), Paragraph("77.41%", table_cell_center), Paragraph("81.13%", table_cell_center), Paragraph("77.79%", table_cell_center)],
        [Paragraph("231", table_cell_bold), Paragraph("99.87%", table_cell_center), Paragraph("66.65%", table_cell_center), Paragraph("50.00%", table_cell_center), Paragraph("55.54%", table_cell_center)],
        [Paragraph("232", table_cell_bold), Paragraph("22.58%", table_cell_center), Paragraph("61.23%", table_cell_center), Paragraph("50.14%", table_cell_center), Paragraph("18.63%", table_cell_center)],
        [Paragraph("233", table_cell_bold), Paragraph("92.81%", table_cell_center), Paragraph("49.86%", table_cell_center), Paragraph("55.25%", table_cell_center), Paragraph("47.49%", table_cell_center)],
        [Paragraph("234", table_cell_bold), Paragraph("98.22%", table_cell_center), Paragraph("99.41%", table_cell_center), Paragraph("67.33%", table_cell_center), Paragraph("67.67%", table_cell_center)],

        [Paragraph("<b>Average (Mean)</b>", table_cell_bold), Paragraph("<b>90.01%</b>", table_cell_center_bold), Paragraph("<b>67.18%</b>", table_cell_center_bold), Paragraph("<b>67.30%</b>", table_cell_center_bold), Paragraph("<b>62.05%</b>", table_cell_center_bold)],
        [Paragraph("<b>Std Deviation</b>", table_cell_bold), Paragraph("<b>&plusmn; 20.23%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 22.20%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 22.66%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 22.68%</b>", table_cell_center_bold)],
        [Paragraph("<b>Mean &plusmn; Std Dev</b>", table_cell_bold), Paragraph("<b>90.01 &plusmn; 20.23%</b>", table_cell_center_bold), Paragraph("<b>67.18 &plusmn; 22.20%</b>", table_cell_center_bold), Paragraph("<b>67.30 &plusmn; 22.66%</b>", table_cell_center_bold), Paragraph("<b>62.05 &plusmn; 22.68%</b>", table_cell_center_bold)]
    ]

    t1 = Table(t1_rows, colWidths=[110, 105, 110, 105, 110])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -4), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BACKGROUND", (0, -3), (-1, -3), colors.HexColor("#EDF2F7")),
        ("BACKGROUND", (0, -2), (-1, -2), colors.HexColor("#EDF2F7")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EBF8FF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
    ]))
    story.append(t1)

    story.append(PageBreak())

    # =======================================================================
    # PAGE 2: TABLE 2 - DS2 PATIENT-WISE MICRO / PER-CLASS METRICS
    # =======================================================================
    story.append(Paragraph("DS2 Independent Test Benchmark: Micro / Per-Class Sensitivity", title_style))
    story.append(Paragraph("Sensitivity (Recall %) across 5 AAMI Heartbeat Classes [N, S, V, F, Q]", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor("#1A365D"), spaceAfter=8))

    story.append(Paragraph("Table 2: DS2 Patient-Wise Micro Sensitivity across [N, S, V, F, Q]", h1_style))

    t2_rows = [
        [Paragraph("Patient Record", table_header_style),
         Paragraph("Normal (N)", table_header_style),
         Paragraph("SVEB (S)", table_header_style),
         Paragraph("VEB (V)", table_header_style),
         Paragraph("Fusion (F)", table_header_style),
         Paragraph("Unknown (Q)", table_header_style)],

        [Paragraph("100", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("69.70%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("103", table_cell_bold), Paragraph("99.95%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("105", table_cell_bold), Paragraph("95.36%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("73.17%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("0.00%", table_cell_center)],
        [Paragraph("111", table_cell_bold), Paragraph("99.95%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("113", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("33.33%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("117", table_cell_bold), Paragraph("33.62%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("121", table_cell_bold), Paragraph("98.33%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("123", table_cell_bold), Paragraph("99.93%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("200", table_cell_bold), Paragraph("98.51%", table_cell_center), Paragraph("60.00%", table_cell_center), Paragraph("98.18%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("202", table_cell_bold), Paragraph("98.83%", table_cell_center), Paragraph("63.64%", table_cell_center), Paragraph("94.74%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("210", table_cell_bold), Paragraph("98.02%", table_cell_center), Paragraph("9.09%", table_cell_center), Paragraph("88.21%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("212", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("213", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("95.00%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("214", table_cell_bold), Paragraph("96.35%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("80.86%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("0.00%", table_cell_center)],
        [Paragraph("219", table_cell_bold), Paragraph("92.79%", table_cell_center), Paragraph("14.29%", table_cell_center), Paragraph("96.88%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("221", table_cell_bold), Paragraph("99.56%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("222", table_cell_bold), Paragraph("79.89%", table_cell_center), Paragraph("73.68%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("228", table_cell_bold), Paragraph("99.11%", table_cell_center), Paragraph("66.67%", table_cell_center), Paragraph("77.62%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("231", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("50.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("232", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("0.29%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("233", table_cell_bold), Paragraph("98.25%", table_cell_center), Paragraph("42.86%", table_cell_center), Paragraph("79.88%", table_cell_center), Paragraph("0.00%", table_cell_center), Paragraph("---", table_cell_center)],
        [Paragraph("234", table_cell_bold), Paragraph("100.00%", table_cell_center), Paragraph("2.00%", table_cell_center), Paragraph("100.00%", table_cell_center), Paragraph("---", table_cell_center), Paragraph("---", table_cell_center)],

        [Paragraph("<b>Average (Mean)</b>", table_cell_bold), Paragraph("<b>94.93%</b>", table_cell_center_bold), Paragraph("<b>33.47%</b>", table_cell_center_bold), Paragraph("<b>89.66%</b>", table_cell_center_bold), Paragraph("<b>0.00%</b>", table_cell_center_bold), Paragraph("<b>0.00%</b>", table_cell_center_bold)],
        [Paragraph("<b>Std Deviation</b>", table_cell_bold), Paragraph("<b>&plusmn; 14.06%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 33.26%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 13.66%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 0.00%</b>", table_cell_center_bold), Paragraph("<b>&plusmn; 0.00%</b>", table_cell_center_bold)],
        [Paragraph("<b>Mean &plusmn; Std Dev</b>", table_cell_bold), Paragraph("<b>94.93 &plusmn; 14.06%</b>", table_cell_center_bold), Paragraph("<b>33.47 &plusmn; 33.26%</b>", table_cell_center_bold), Paragraph("<b>89.66 &plusmn; 13.66%</b>", table_cell_center_bold), Paragraph("<b>0.00 &plusmn; 0.00%</b>", table_cell_center_bold), Paragraph("<b>0.00 &plusmn; 0.00%</b>", table_cell_center_bold)]
    ]

    t2 = Table(t2_rows, colWidths=[100, 88, 88, 88, 88, 88])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -4), [colors.white, colors.HexColor("#F8FAFC")]),
        ("BACKGROUND", (0, -3), (-1, -3), colors.HexColor("#EDF2F7")),
        ("BACKGROUND", (0, -2), (-1, -2), colors.HexColor("#EDF2F7")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EBF8FF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
    ]))
    story.append(t2)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] PDF successfully generated: {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    create_pdf()
