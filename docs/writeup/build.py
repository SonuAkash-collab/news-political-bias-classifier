"""Build docs/writeup/writeup.pdf from docs/writeup/writeup.md.

Ensures the final write-up is strictly at most 2 pages as required by the
assignment instructions.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = Path(__file__).resolve().parent
MD_PATH = DOCS_DIR / "writeup.md"
PDF_PATH = DOCS_DIR / "writeup.pdf"

# Palette definitions
MIDNIGHT = colors.HexColor("#051218")
DEEP_TEAL = colors.HexColor("#053944")
OCEAN_TEAL = colors.HexColor("#0A5C68")
TURQUOISE = colors.HexColor("#088192")
GOLD = colors.HexColor("#BCAC79")
LIGHT_BG = colors.HexColor("#F8F7F4")
MUTED = colors.HexColor("#5B6B70")
RULE_COLOR = colors.HexColor("#D0CBC0")


def parse_markdown(text: str):
    """Simple parser for writeup.md converting sections into ReportLab elements."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        textColor=MIDNIGHT,
        spaceAfter=2,
    )
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=MUTED,
    )
    h2_style = ParagraphStyle(
        "Heading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=11.5,
        textColor=DEEP_TEAL,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.3,
        leading=9.4,
        textColor=MIDNIGHT,
        spaceAfter=3.5,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.4,
        leading=7.8,
        textColor=MIDNIGHT,
    )
    table_head_style = ParagraphStyle(
        "TableHead",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.4,
        leading=7.8,
        textColor=colors.white,
    )

    story = []

    lines = text.strip().split("\n")
    i = 0
    in_table = False
    table_rows = []

    while i < len(lines):
        line = lines[i].rstrip()

        # Title
        if line.startswith("# "):
            title_text = line[2:].strip()
            story.append(Paragraph(title_text, title_style))
            i += 1
            continue

        # Metadata block
        if line.startswith("**Team 16") or line.startswith("**Course:") or line.startswith("**Reference"):
            clean_line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(clean_line, meta_style))
            i += 1
            continue

        # Heading 2
        if line.startswith("## "):
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name == "MetaText":
                story.append(Spacer(1, 2))
                story.append(HRFlowable(width="100%", thickness=0.6, color=RULE_COLOR, spaceBefore=2, spaceAfter=4))
            heading_text = line[3:].strip()
            story.append(Paragraph(heading_text, h2_style))
            i += 1
            continue

        # Markdown Table
        if line.startswith("|"):
            # Check if separator row
            if "---|---" in line:
                i += 1
                continue
            cols = [c.strip() for c in line.strip("|").split("|")]
            table_rows.append(cols)
            # Peek if next line is table
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                continue
            else:
                # End of table, build Table Flowable
                num_cols = len(table_rows[0])
                data = []
                for row_idx, row in enumerate(table_rows):
                    row_cells = []
                    for col_idx, cell in enumerate(row):
                        st = table_head_style if row_idx == 0 else table_cell_style
                        # Replace bold tags
                        cell_fmt = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", cell)
                        row_cells.append(Paragraph(cell_fmt, st))
                    data.append(row_cells)

                # Column widths for 5 columns totaling 540 pt (page width 612 - 2*36 = 540)
                col_widths = [165, 120, 60, 60, 135]
                t = Table(data, colWidths=col_widths)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), DEEP_TEAL),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 1.6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE_COLOR),
                ]))
                story.append(Spacer(1, 2))
                story.append(t)
                story.append(Spacer(1, 3))
                table_rows = []
                i += 1
                continue

        # Normal paragraph or numbered list item
        if line.strip():
            # Format bold text
            fmt_line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(fmt_line, body_style))
        i += 1

    return story


def build_pdf(md_path: Path = MD_PATH, out_path: Path = PDF_PATH) -> int:
    """Compile writeup.md into writeup.pdf, returning page count."""
    text = md_path.read_text(encoding="utf-8")
    story = parse_markdown(text)

    # 0.5 in margins (36 pt)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )
    doc.build(story)

    # Verify page count with PyMuPDF
    pdf_doc = fitz.open(str(out_path))
    num_pages = len(pdf_doc)
    print(f"Built {out_path.name}: {num_pages} page(s).")
    return num_pages


if __name__ == "__main__":
    pages = build_pdf()
    if pages > 2:
        print(f"ERROR: PDF exceeds 2 pages (currently {pages} pages)!", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"SUCCESS: PDF is exactly {pages} page(s) (<= 2 required).")
