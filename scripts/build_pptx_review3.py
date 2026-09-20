"""Build `Review3_19MIS1127.pptx` from Review 3 slide content and speaker notes.

Authority & Provenance
---------------------
Review 3 (Guide only: Dr. M. Premalatha, 30 marks).
Registered VTOP title (verbatim):
"An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease Detection Using Image"

Renders `docs/presentation/ppt_content_review3.md` with notes from
`docs/presentation/speaker_notes_review3.md`.
Output: `Review3_19MIS1127.pptx`.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

NEWLINE = chr(10)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT = PROJECT_ROOT / "docs" / "presentation" / "ppt_content_review3.md"
NOTES = PROJECT_ROOT / "docs" / "presentation" / "speaker_notes_review3.md"

# 16:9 widescreen layout
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x0B, 0x4F, 0x8A)      # heading blue
MUTED = RGBColor(0x5A, 0x5A, 0x5A)       # source lines
RULE = RGBColor(0xD8, 0xDD, 0xE2)
HEADER_FILL = RGBColor(0xEC, 0xF2, 0xF8)

MARGIN = Inches(0.55)
TITLE_H = Inches(0.95)

PRESENTER_MARKERS = (
    "read this table before the review", "read this before the review", "say this out loud",
    "do not \"correct\" it", "be honest about", "if challenged", "never say",
    "what to say", "do not read", "verification note",
)


@dataclass
class Block:
    kind: str                      # heading | para | bullets | numbers | table | figure | code | quote
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    path: str = ""
    start: int = 1


@dataclass
class Slide:
    number: str
    title: str
    blocks: list[Block] = field(default_factory=list)
    source: str = ""
    notes: str = ""


# --- LaTeX safety net -------------------------------------------------------
# The slide source is written in Unicode, not LaTeX. This map exists so that a
# macro that survives an edit renders as a symbol on the projector instead of
# as raw "\pm" — the failure mode that put 58 unreadable lines on the deck
# before Review 3. It is a fallback, not the intended input format.
LATEX_SYMBOLS = {
    "pm": "±", "mp": "∓", "times": "×", "div": "÷", "cdot": "·", "ast": "*",
    "approx": "≈", "sim": "~", "simeq": "≃", "equiv": "≡", "propto": "∝",
    "neq": "≠", "ne": "≠", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "ll": "≪", "gg": "≫", "rightarrow": "→", "to": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "leftrightarrow": "↔", "mapsto": "↦",
    "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆", "cup": "∪",
    "cap": "∩", "forall": "∀", "exists": "∃", "emptyset": "∅",
    "sum": "∑", "prod": "∏", "int": "∫", "sqrt": "√", "infty": "∞",
    "partial": "∂", "nabla": "∇", "angle": "∠", "perp": "⊥", "circ": "°",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    "ldots": "…", "dots": "…", "cdots": "…", "hline": "", "quad": " ",
    "qquad": "  ", "bigl": "", "bigr": "", "Bigl": "", "Bigr": "",
    "left": "", "right": "", "displaystyle": "", "limits": "", "nolimits": "",
}
_LATEX_NAME_RE = re.compile(
    r"\\(" + "|".join(sorted(LATEX_SYMBOLS, key=len, reverse=True)) + r")(?![A-Za-z])"
)
# Wrappers whose braces carry no meaning once the maths is plain text.
_LATEX_WRAPPER_RE = re.compile(
    r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathcal|mathbb|"
    r"operatorname|hat|bar|tilde|vec|overline|underline)\s*\{([^{}]*)\}"
)
_SUPERSCRIPT = str.maketrans("0123456789+-n()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ⁽⁾")


def _superscript(match: re.Match) -> str:
    body = match.group(1) or match.group(2) or ""
    if body and all(c in "0123456789+-n()" for c in body):
        return body.translate(_SUPERSCRIPT)
    return "^" + body


def _latex_to_unicode(text: str) -> str:
    """Render leftover LaTeX as Unicode. Applied only when a backslash is present."""
    for _ in range(3):                                   # resolve nested wrappers
        replaced = _LATEX_WRAPPER_RE.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced
    text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1) / (\2)", text)
    text = _LATEX_NAME_RE.sub(lambda m: LATEX_SYMBOLS[m.group(1)], text)
    text = re.sub(r"\\[,;:!]", " ", text)
    text = text.replace("\\\\", " ").replace("\\|", "|").replace("\\%", "%")
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)         # unknown macro: keep the word
    return text.replace("{", "").replace("}", "")


def _strip_math(text: str) -> str:
    """Drop maths delimiters while leaving a currency dollar sign intact."""
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.S)
    text = re.sub(r"\$([^$\n]*[\\^_=][^$\n]*)\$", r"\1", text)
    return re.sub(r"\$(?!\d)", "", text)


def _strip_md(text: str) -> str:
    """Markdown to plain text."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"~~([^~]*)~~", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    if "$" in text:                                      # delimiters first, while
        text = _strip_math(text)                         # the macros still mark maths
    if "\\" in text:
        text = _latex_to_unicode(text)
    text = re.sub(r"\^\{([^{}]*)\}|\^([0-9+\-n]+)", _superscript, text)
    return text.replace("&nbsp;", " ").strip()


def _is_presenter_note(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in PRESENTER_MARKERS)


BULLET_RE = r"[-*]\s+(.*)"
NUMBER_RE = r"\d+\.\s+(.*)"


def parse_content(markdown: str) -> list[Slide]:
    parts = re.split(r"^## (Slide [^\r\n]*)$", markdown, flags=re.M)
    slides: list[Slide] = []

    for i in range(1, len(parts), 2):
        heading, body = parts[i].strip(), parts[i + 1]

        match = re.match(r"Slide\s+(A?\d+)[^—\-–]*[—\-–]\s*(.*)", heading)
        if not match:
            continue
        title = _strip_md(match.group(2))
        if match.group(1).startswith("A"):
            title = "Appendix — " + title
        slide = Slide(number=match.group(1), title=title)

        lines = body.splitlines()
        pending: list[str] = []
        note_lines: list[str] = []

        def flush_para() -> None:
            if not pending:
                return
            text = " ".join(pending).strip()
            pending.clear()
            if not text:
                return
            if _is_presenter_note(text):
                note_lines.append(text)
            else:
                slide.blocks.append(Block("para", _strip_md(text)))

        index = 0
        while index < len(lines):
            line = lines[index].rstrip()
            stripped = line.strip()

            if not stripped or stripped == "---":
                flush_para()
                index += 1
                continue

            if stripped.startswith("```"):
                flush_para()
                index += 1
                code_lines = []
                while index < len(lines) and not lines[index].strip().startswith("```"):
                    code_lines.append(lines[index].rstrip())
                    index += 1
                index += 1
                while code_lines and not code_lines[0].strip():
                    code_lines.pop(0)
                while code_lines and not code_lines[-1].strip():
                    code_lines.pop()
                if code_lines:
                    slide.blocks.append(Block("code", text=NEWLINE.join(code_lines)))
                continue

            figure = re.match(r"(?:\*\*\[|\[)?(FIGURES?|SCREENSHOT):\s*([^\]]+)(?:\]\*\*)?", stripped, re.IGNORECASE)
            if figure:
                flush_para()
                for candidate in re.split(r"\s*[·,]\s*|\s+and\s+", figure.group(2)):
                    candidate = candidate.strip().strip("`")
                    if not candidate:
                        continue
                    if "/" not in candidate:
                        candidate = "reports/figures/" + candidate
                    slide.blocks.append(Block("figure", path=candidate))
                index += 1
                continue

            if "reports/figures/" in stripped and ".png" in stripped:
                paths = re.findall(r"reports/figures/[\w_-]+\.png", stripped)
                if paths:
                    flush_para()
                    for p in paths:
                        slide.blocks.append(Block("figure", path=p))
                    index += 1
                    continue

            if stripped.startswith("**Traces to:**"):
                flush_para()
                slide.source = _strip_md(stripped.replace("**Traces to:**", "").strip())
                index += 1
                continue

            if stripped.startswith("|"):
                flush_para()
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [c.strip() for c in lines[index].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        rows.append([_strip_md(c) for c in cells])
                    index += 1
                if rows:
                    slide.blocks.append(Block("table", rows=rows))
                continue

            if stripped.startswith(">"):
                flush_para()
                quote = []
                while index < len(lines) and lines[index].strip().startswith(">"):
                    quote.append(lines[index].strip().lstrip("> ").strip())
                    index += 1
                text = " ".join(q for q in quote if q).strip()
                if text.startswith("#"):
                    slide.blocks.append(Block("heading", _strip_md(text.lstrip("# "))))
                elif _is_presenter_note(text):
                    note_lines.append(text)
                elif text:
                    slide.blocks.append(Block("quote", _strip_md(text)))
                continue

            if stripped.startswith("### ") or stripped.startswith("## ") or stripped.startswith("# "):
                flush_para()
                slide.blocks.append(Block("heading", _strip_md(stripped.lstrip("# "))))
                index += 1
                continue

            bullet = re.match(r"[-*]\s+(.*)", stripped)
            number = re.match(r"\d+\.\s+(.*)", stripped)
            if bullet or number:
                flush_para()
                kind = "bullets" if bullet else "numbers"
                items = []
                while index < len(lines):
                    raw = lines[index]
                    current = raw.strip()
                    match_item = (re.match(BULLET_RE, current) if bullet
                                  else re.match(NUMBER_RE, current))
                    if match_item:
                        items.append(_strip_md(match_item.group(1)))
                        index += 1
                    elif items and raw.startswith(' ') and current:
                        items[-1] += ' ' + _strip_md(current)
                        index += 1
                    else:
                        break
                items = [_strip_md(item) for item in items]
                slide.blocks.append(Block(kind, items=items))
                continue

            pending.append(stripped)
            index += 1

        flush_para()

        has_visual = any(b.kind in ("table", "figure") for b in slide.blocks)
        if has_visual:
            kept = []
            for block in slide.blocks:
                if block.kind in ("para", "quote") and len(block.text) > 140:
                    note_lines.append(block.text)
                else:
                    kept.append(block)
            slide.blocks = kept
        slide.notes = _drop_anticipated_questions(NEWLINE.join(note_lines))
        slides.append(slide)

    return slides


QUESTION_HEADINGS = (
    "Possible Questions", "Defensible Answers", "Possible Question",
    "Defensible Answer", "Questions you will be asked",
    "Things to admit immediately if asked",
)


def _drop_anticipated_questions(text: str) -> str:
    kept, skipping = [], False
    for line in text.split(NEWLINE):
        stripped = line.strip()
        if any(stripped.startswith(h) for h in QUESTION_HEADINGS):
            skipping = True
            continue
        if skipping and re.match(r"^[A-Z][A-Za-z /]{3,40}[.]\s", stripped):
            skipping = False
        if not skipping:
            kept.append(line)
    out = NEWLINE.join(kept)
    return re.sub(r"\n{3,}", NEWLINE * 2, out).strip()


def _rejoin_wrapped_emphasis(text: str) -> str:
    out = []
    for line in text.splitlines():
        if out and out[-1].count("**") % 2 == 1 and line.strip():
            out[-1] = out[-1].rstrip() + " " + line.strip()
        else:
            out.append(line)
    return NEWLINE.join(out)


def _strip_md_block(text: str) -> str:
    out = []
    for line in _rejoin_wrapped_emphasis(text).splitlines():
        line = line.strip()
        if line in ("---", ""):
            out.append("")
            continue
        out.append(_strip_md(line.lstrip("> ")))
    return "\n".join(out)


def parse_notes(markdown: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = re.split(r"^# (Slide[^\r\n]*)$", markdown, flags=re.M)
    for i in range(1, len(parts), 2):
        heading, body = parts[i], parts[i + 1]
        span = re.search(r"^Slide\s+(\d+)\s*[–-]\s*(\d+)", heading)
        single = re.search(r"^Slide\s+(A?\d+)", heading)
        if span:
            numbers = [str(n) for n in range(int(span.group(1)), int(span.group(2)) + 1)]
        elif single:
            numbers = [single.group(1)]
        else:
            numbers = re.findall(r"A?\d+", heading)
        if not numbers:
            continue
        text = re.sub(r"\n{3,}", "\n\n", _strip_md_block(body)).strip()
        text = _drop_anticipated_questions(text)
        for number in numbers:
            out.setdefault(number, text)
    return out


# Layout constants
CONTENT_TOP = Inches(1.28)
CONTENT_BOTTOM = Inches(6.92)
CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP
GUTTER = Inches(0.28)
BLOCK_GAP = Inches(0.12)
TEXT_SHARE = 0.55

BODY_PT, TABLE_PT = 14, 10.5


def _content_width():
    return SLIDE_W - 2 * MARGIN


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Emu(0)
    frame.margin_top = frame.margin_bottom = Emu(0)
    return frame


def _para(frame, text: str, size: int | float, *, bold=False, italic=False, colour=INK,
          space_after=5, bullet=False, first=False):
    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
    paragraph.text = ("•  " + text) if bullet else text
    paragraph.space_after = Pt(space_after)
    paragraph.line_spacing = 1.15
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = colour
        run.font.name = "Calibri"
    return paragraph


def _lines_needed(text: str, width_in: float, size: int | float) -> int:
    chars_per_line = max(12, int(width_in * 72 / (size * 0.48)))
    return max(1, -(-len(text) // chars_per_line))


def _block_height(block, width_in: float, body: int, table_font: int | float) -> float:
    line = body * 1.22 / 72
    if block.kind == "table":
        n_cols = len(block.rows[0])
        widths = [width_in * (0.30 if n_cols >= 4 else 0.36)] if n_cols > 2 else [width_in / n_cols]
        widths += [(width_in - widths[0]) / (n_cols - 1)] * (n_cols - 1) if n_cols > 1 else []
        cell_line = table_font * 1.12 / 72
        total = 0.0
        for row in block.rows:
            lines = 1
            for c, cell in enumerate(row[:n_cols]):
                usable = max(0.4, widths[min(c, len(widths) - 1)] - 0.14)
                per_line = max(6, int(usable * 72 / (table_font * 0.48)))
                lines = max(lines, -(-len(cell) // per_line))
            total += max(0.22, lines * cell_line + 0.05)
        return total + 0.12
    if block.kind == "code":
        return 0.020 * 10 * len(block.text.split(NEWLINE)) + 0.16
    if block.kind in ("bullets", "numbers"):
        return sum(line * _lines_needed(i, width_in, body) + 0.028 for i in block.items) + 0.05
    if block.kind == "heading":
        return line * 1.2 + 0.08
    return line * _lines_needed(block.text, width_in, body) + 0.08


def _estimate_height(blocks, width_in: float, body: int, table_font: int | float) -> float:
    return sum(_block_height(b, width_in, body, table_font)
               for b in blocks if b.kind != "figure")


def _is_reference_list(title: str) -> bool:
    return title.lower().startswith("references") or "references" in title.lower()


def _pick_sizes(blocks, width_in: float) -> tuple[int, float]:
    budget = CONTENT_H / Inches(1) * 1.08
    for table_font in (TABLE_PT, 10, 9.5, 9, 8.5):
        for body in (BODY_PT, 13, 12, 11):
            if _estimate_height(blocks, width_in, body, table_font) <= budget:
                return body, table_font
    return 11, 8.5


def _title_bar(slide, text: str, number: str) -> None:
    bar = slide.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, Inches(1.00))
    bar.fill.solid()
    bar.fill.fore_color.rgb = HEADER_FILL
    bar.line.fill.background()
    bar.shadow.inherit = False

    accent = slide.shapes.add_shape(1, Emu(0), Inches(1.00), SLIDE_W, Inches(0.035))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()
    accent.shadow.inherit = False

    frame = _textbox(slide, MARGIN, Inches(0.14), _content_width() - Inches(1.0), Inches(0.74))
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    size = 24 if len(text) < 55 else (20 if len(text) < 85 else 17)
    _para(frame, text, size, bold=True, colour=ACCENT, space_after=0, first=True)

    tag = _textbox(slide, SLIDE_W - MARGIN - Inches(1.0), Inches(0.14), Inches(1.0), Inches(0.74))
    tag.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = _para(tag, f"Slide {number}", 12, bold=True, colour=MUTED, space_after=0, first=True)
    paragraph.alignment = PP_ALIGN.RIGHT


def _add_table(slide, rows, left, top, width, font_size: int | float):
    n_rows, n_cols = len(rows), len(rows[0])
    row_h = Inches(0.26 if font_size >= 10 else 0.23)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, row_h * n_rows)
    table = shape.table

    if n_cols > 1:
        weights = []
        for c in range(n_cols):
            longest = max((len(row[c]) for row in rows if c < len(row)), default=4)
            average = sum(len(row[c]) for row in rows if c < len(row)) / max(1, len(rows))
            weights.append(max(4.0, (longest + 2 * average) / 3))
        total = sum(weights)
        floor = 0.55 * 914400
        widths = [max(floor, width * w / total) for w in weights]
        overflow = sum(widths) - width
        if overflow > 0:
            order = sorted(range(n_cols), key=lambda c: widths[c], reverse=True)
            for c in order:
                take = min(overflow, widths[c] - floor)
                widths[c] -= take
                overflow -= take
                if overflow <= 0:
                    break
        for c in range(n_cols):
            table.columns[c].width = Emu(int(widths[c]))

    for r, row in enumerate(rows):
        table.rows[r].height = row_h
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.text = row[c] if c < len(row) else ""
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEADER_FILL if r == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.space_after = Pt(0)
            paragraph.line_spacing = 1.08
            for run in paragraph.runs:
                run.font.size = Pt(font_size)
                run.font.name = "Calibri"
                run.font.bold = (r == 0)
                run.font.color.rgb = ACCENT if r == 0 else INK
    return shape, row_h * n_rows


def _place_figures(slide, figures, left, top, width, height) -> None:
    from PIL import Image

    present = [f for f in figures if (PROJECT_ROOT / f.path).exists()][:2]
    if not present:
        return
    slot_h = (height - BLOCK_GAP * (len(present) - 1)) / len(present)

    for index, figure in enumerate(present):
        path = PROJECT_ROOT / figure.path
        with Image.open(path) as image:
            aspect = image.height / image.width
        draw_w = width
        draw_h = Emu(int(draw_w * aspect))
        if draw_h > slot_h:
            draw_h = Emu(int(slot_h))
            draw_w = Emu(int(draw_h / aspect))
        slot_top = top + (slot_h + BLOCK_GAP) * index
        slide.shapes.add_picture(
            str(path),
            Emu(int(left + (width - draw_w) / 2)),
            Emu(int(slot_top + (slot_h - draw_h) / 2)),
            width=Emu(int(draw_w)), height=Emu(int(draw_h)),
        )


def render_title_slide(prs, slide_data) -> None:
    """Render Slide 1: exact registered title, candidate and guide metadata cards, and institutional block."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    # Top accent band
    band = slide.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, Inches(0.18))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False

    # Exact registered title (VTOP registered verbatim)
    REGISTERED_TITLE = "An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease Detection Using Image"
    heading = next((b.text for b in slide_data.blocks if "shrimp" in b.text.lower() and len(b.text) > 30), REGISTERED_TITLE)
    frame = _textbox(slide, MARGIN, Inches(0.85), _content_width(), Inches(1.35))
    paragraph = _para(frame, heading, 25, bold=True, colour=ACCENT, space_after=0, first=True)
    paragraph.line_spacing = 1.18

    # Subtitle / context line
    frame_sub = _textbox(slide, MARGIN, Inches(2.28), _content_width(), Inches(0.36))
    _para(frame_sub, "Review 3 — Final Project Review (Guide Evaluation: 30 Marks)",
          13.5, bold=True, colour=MUTED, space_after=0, first=True)

    # Subtle divider rule
    rule = slide.shapes.add_shape(1, MARGIN, Inches(2.78), _content_width(), Emu(9525))
    rule.fill.solid()
    rule.fill.fore_color.rgb = RULE
    rule.line.fill.background()
    rule.shadow.inherit = False

    # Two parallel highlight cards for Candidate and Guide
    card_w = (_content_width() - Inches(0.40)) / 2
    card_top = Inches(3.12)
    card_h = Inches(1.95)

    # Left Card: Candidate
    card_l = slide.shapes.add_shape(1, MARGIN, card_top, card_w, card_h)
    card_l.fill.solid()
    card_l.fill.fore_color.rgb = HEADER_FILL
    card_l.line.color.rgb = RULE
    card_l.line.width = Pt(1)
    card_l.shadow.inherit = False

    frame_l = _textbox(slide, MARGIN + Inches(0.28), card_top + Inches(0.22),
                       card_w - Inches(0.56), card_h - Inches(0.44))
    _para(frame_l, "PROJECT CANDIDATE", 11, bold=True, colour=MUTED, space_after=3, first=True)
    _para(frame_l, "G Mohan Jayanth", 18, bold=True, colour=INK, space_after=3)
    _para(frame_l, "Register No: 19MIS1127", 13, bold=False, colour=INK, space_after=2)
    _para(frame_l, "M.Tech Integrated Software Engineering (5-Year)", 12, bold=False, colour=MUTED, space_after=0)

    # Right Card: Guide
    right_x = MARGIN + card_w + Inches(0.40)
    card_r = slide.shapes.add_shape(1, right_x, card_top, card_w, card_h)
    card_r.fill.solid()
    card_r.fill.fore_color.rgb = HEADER_FILL
    card_r.line.color.rgb = RULE
    card_r.line.width = Pt(1)
    card_r.shadow.inherit = False

    frame_r = _textbox(slide, right_x + Inches(0.28), card_top + Inches(0.22),
                       card_w - Inches(0.56), card_h - Inches(0.44))
    _para(frame_r, "PROJECT GUIDE & SUPERVISOR", 11, bold=True, colour=MUTED, space_after=3, first=True)
    _para(frame_r, "Dr. M. Premalatha", 18, bold=True, colour=INK, space_after=3)
    _para(frame_r, "Professor & Head of Department", 13, bold=False, colour=INK, space_after=2)
    _para(frame_r, "School of Computer Science and Engineering (SCOPE)", 12, bold=False, colour=MUTED, space_after=0)

    # Institutional Block
    frame_inst = _textbox(slide, MARGIN, Inches(5.35), _content_width(), Inches(1.10))
    _para(frame_inst, "School of Computer Science and Engineering (SCOPE)",
          14, bold=True, colour=ACCENT, space_after=2, first=True)
    _para(frame_inst, "Vellore Institute of Technology, Chennai Campus",
          13, bold=True, colour=INK, space_after=4)
    _para(frame_inst, "Academic Year 2025–2026   ·   Review 3 (Final Guide Review: 30 Marks)   ·   September 2026",
          11, bold=False, colour=MUTED, space_after=0)

    # Footer / trace line
    footer = _textbox(slide, MARGIN, Inches(6.85), _content_width(), Inches(0.30))
    _para(footer, "Registered VTOP Title (Verbatim)   ·   Review 3 (Final Guide Review: 30 Marks)   ·   Commitment Matrix: Appendix Slide A1",
          10, colour=MUTED, space_after=0, first=True)


def _clean_notes(text: str) -> str:
    return _strip_md(_rejoin_wrapped_emphasis(text)).replace("**", "").replace("`", "")


def render_slide(prs, slide_data) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title_bar(slide, slide_data.title, slide_data.number)

    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    has_figure = any((PROJECT_ROOT / f.path).exists() for f in figures)

    content_top = CONTENT_TOP
    content_bottom = CONTENT_BOTTOM
    content_height = content_bottom - content_top

    if has_figure:
        text_w = Emu(int(_content_width() * TEXT_SHARE))
        fig_left = MARGIN + text_w + GUTTER
        fig_w = _content_width() - text_w - GUTTER
        _place_figures(slide, figures, fig_left, content_top, fig_w, content_height)
        render_w = text_w
    else:
        render_w = _content_width()

    width_in = render_w / Inches(1)
    body_pt, table_pt = _pick_sizes(others, width_in)
    if _is_reference_list(slide_data.title):
        body_pt = 11

    top = content_top
    for block in others:
        if block.kind == "table" and block.rows:
            _, drawn_h = _add_table(slide, block.rows, MARGIN, top, render_w, table_pt)
            top += drawn_h + BLOCK_GAP
        elif block.kind == "code":
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            lines = block.text.split(NEWLINE)
            for i, line in enumerate(lines):
                p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
                p.text = line
                p.space_after = Pt(0)
                p.line_spacing = 1.05
                for run in p.runs:
                    run.font.name = "Consolas"
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = INK
            top += h + BLOCK_GAP
        elif block.kind == "heading":
            h = Inches(0.36)
            frame = _textbox(slide, MARGIN, top, render_w, h)
            _para(frame, block.text, body_pt + 1.5, bold=True, colour=ACCENT, space_after=3, first=True)
            top += h + Inches(0.04)
        elif block.kind in ("bullets", "numbers"):
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            for i, item in enumerate(block.items):
                is_bullet = (block.kind == "bullets")
                prefix = "" if is_bullet else f"{i + block.start}. "
                _para(frame, prefix + item, body_pt, bullet=is_bullet, space_after=4, first=(i == 0))
            top += h + BLOCK_GAP
        elif block.kind in ("para", "quote"):
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            italic = (block.kind == "quote")
            _para(frame, block.text, body_pt, italic=italic, space_after=5, first=True)
            top += h + BLOCK_GAP

    if slide_data.source:
        footer = _textbox(slide, MARGIN, Inches(7.02), _content_width(), Inches(0.30))
        _para(footer, f"Traces to: {slide_data.source}", 9.5, colour=MUTED, space_after=0, first=True)

    if slide_data.notes:
        slide.notes_slide.notes_text_frame.text = _clean_notes(slide_data.notes)


def _paginate(slide_data: Slide) -> list[Slide]:
    """Check if content fits on a single slide; if not, split gracefully."""
    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    has_figure = any((PROJECT_ROOT / f.path).exists() for f in figures)

    render_w = _content_width() * TEXT_SHARE if has_figure else _content_width()
    width_in = render_w / Inches(1)

    # Budget in inches
    budget = CONTENT_H / Inches(1) * 1.08
    body_pt, table_pt = _pick_sizes(others, width_in)
    if _is_reference_list(slide_data.title):
        body_pt = 11

    total_h = _estimate_height(others, width_in, body_pt, table_pt)
    if total_h <= budget or len(others) <= 1:
        return [slide_data]

    # Split into multiple pages
    pages: list[list[Block]] = []
    current: list[Block] = []
    current_h = 0.0

    for block in others:
        bh = _block_height(block, width_in, body_pt, table_pt)
        if current and (current_h + bh > budget):
            pages.append(current)
            current = [block]
            current_h = bh
        else:
            current.append(block)
            current_h += bh
    if current:
        pages.append(current)

    out = []
    for i, page in enumerate(pages):
        title = slide_data.title if i == 0 else f"{slide_data.title} (cont.)"
        part = Slide(
            number=slide_data.number if i == 0 else f"{slide_data.number}b",
            title=title,
            blocks=(figures + page) if i == 0 else page,
            source=slide_data.source if i == len(pages) - 1 else "",
            notes=(slide_data.notes if i == 0
                   else f"Continued from slide {slide_data.number}. The speaker notes for this material are on that slide."),
        )
        out.append(part)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="Review3_19MIS1127.pptx")
    args = parser.parse_args(argv)

    slides = parse_content(CONTENT.read_text(encoding="utf-8"))
    notes = parse_notes(NOTES.read_text(encoding="utf-8"))

    for slide in slides:
        extra = notes.get(slide.number, "")
        if extra:
            slide.notes = (slide.notes + "\n\n" + extra).strip() if slide.notes else extra

    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    split = []
    for slide in slides:
        if slide.number == "1":
            render_title_slide(prs, slide)
            if slide.notes:
                prs.slides[-1].notes_slide.notes_text_frame.text = _clean_notes(slide.notes)
            continue
        for part_index, part in enumerate(_paginate(slide)):
            render_slide(prs, part)
            if part_index:
                split.append(slide.number)

    out = PROJECT_ROOT / args.out
    prs.save(out)

    with_notes = sum(1 for s in prs.slides if s.has_notes_slide
                     and s.notes_slide.notes_text_frame.text.strip())
    print(f"wrote {len(prs.slides)} slides -> {out.name}")
    print(f"speaker notes attached to {with_notes} of {len(prs.slides)}")
    if split:
        print()
        print("continued onto a second slide because the content did not fit:")
        for number in dict.fromkeys(split):
            print(f"   slide {number}  ->  '{number}' + '{number}b'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
