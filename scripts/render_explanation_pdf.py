from __future__ import annotations

import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "EXPLANATION.md"
OUTPUT = Path(
    os.getenv(
        "EXPLANATION_PDF_OUTPUT",
        str(ROOT / "output" / "pdf" / "explanation.pdf"),
    )
)


def parse_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    title = lines[0].lstrip("# ").strip()
    sections: list[tuple[str, str]] = []
    for line in lines[1:]:
        match = re.match(r"\*\*(.+?)\.\*\*\s*(.+)", line)
        if not match:
            continue
        sections.append((match.group(1), match.group(2)))
    return title, sections


def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render() -> None:
    title, sections = parse_sections(INPUT.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(OUTPUT), pagesize=letter)
    width, height = letter
    margin = 54
    y = height - margin

    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, y, title)
    y -= 28

    pdf.setStrokeColor(colors.HexColor("#D1D5DB"))
    pdf.line(margin, y, width - margin, y)
    y -= 24

    body_font = "Helvetica"
    body_size = 10
    line_height = 14
    max_width = width - (margin * 2)

    for heading, body in sections:
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(margin, y, f"{heading}.")
        y -= 16

        pdf.setFillColor(colors.HexColor("#1F2937"))
        pdf.setFont(body_font, body_size)
        for line in wrap_text(body, body_font, body_size, max_width):
            if y < margin + 30:
                raise RuntimeError("Explanation text did not fit on one page.")
            pdf.drawString(margin, y, line)
            y -= line_height
        y -= 10

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#6B7280"))
    pdf.drawRightString(width - margin, margin - 18, "RAG QA explanation - one page")
    try:
        pdf.save()
    except PermissionError:
        fallback = OUTPUT.with_name(f"{OUTPUT.stem}_updated{OUTPUT.suffix}")
        pdf = canvas.Canvas(str(fallback), pagesize=letter)
        _render_page(pdf, title, sections)
        pdf.save()
        print(fallback)
        return

    print(OUTPUT)


def _render_page(pdf: canvas.Canvas, title: str, sections: list[tuple[str, str]]) -> None:
    width, height = letter
    margin = 54
    y = height - margin

    pdf.setFillColor(colors.HexColor("#111827"))
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, y, title)
    y -= 28

    pdf.setStrokeColor(colors.HexColor("#D1D5DB"))
    pdf.line(margin, y, width - margin, y)
    y -= 24

    body_font = "Helvetica"
    body_size = 10
    line_height = 14
    max_width = width - (margin * 2)

    for heading, body in sections:
        pdf.setFillColor(colors.HexColor("#111827"))
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(margin, y, f"{heading}.")
        y -= 16

        pdf.setFillColor(colors.HexColor("#1F2937"))
        pdf.setFont(body_font, body_size)
        for line in wrap_text(body, body_font, body_size, max_width):
            if y < margin + 30:
                raise RuntimeError("Explanation text did not fit on one page.")
            pdf.drawString(margin, y, line)
            y -= line_height
        y -= 10

    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#6B7280"))
    pdf.drawRightString(width - margin, margin - 18, "RAG QA explanation - one page")


if __name__ == "__main__":
    render()
