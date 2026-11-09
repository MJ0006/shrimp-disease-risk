"""Build `Review3_19MIS1127.pptx` from Review 3 slide content and speaker notes.

Authority & Provenance
---------------------
Review 3 (Guide only: Dr. M. Premalatha, 30 marks).
Registered VTOP title (verbatim):
"An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp Disease Detection Using Image"

Renders `docs/presentation/ppt_content_review3.md` with notes from
`docs/presentation/speaker_notes_review3.md`, on the design template in
`data/assets/review_template.pptx` (theme, font, title and closing slides).
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
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_THEME_COLOR
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

NEWLINE = chr(10)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTENT = PROJECT_ROOT / "docs" / "presentation" / "ppt_content_review3.md"
NOTES = PROJECT_ROOT / "docs" / "presentation" / "speaker_notes_review3.md"

# The deck is laid out on a content-free template taken from a reference review deck:
# its theme (Savon, with its Avenir Next font swapped for Segoe UI), layouts, title slide and closing slide.
# Only the design is used; every word on the slides comes from CONTENT.
TEMPLATE = PROJECT_ROOT / "data" / "assets" / "review_template.pptx"
TITLE_AND_CONTENT = "Title and Content"

# 16:9 widescreen layout, matching the template
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

MARGIN = Inches(0.85)                      # inside the template's framed panel

PRESENTER_MARKERS = (
    "read this table before the review", "read this before the review", "say this out loud",
    "do not \"correct\" it", "be honest about", "if challenged", "never say",
    "what to say", "do not read", "verification note",
)


@dataclass
class Block:
    kind: str                      # heading | para | bullets | numbers | table | figure | code | quote | flow | lanes
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

        # A letter suffix ("6b") marks a continuation slide, which carries no notes of its own.
        match = re.match(r"Slide\s+(A?\d+[a-z]?)[^—\-–]*[—\-–]\s*(.*)", heading)
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
        flow_next = False          # set by a [DIAGRAM: flow] line; applies to the next table
        lanes_next = False         # set by a [DIAGRAM: lanes] line; applies to the next table
        while index < len(lines):
            line = lines[index].rstrip()
            stripped = line.strip()

            if re.fullmatch(r"(?:\*\*)?\[DIAGRAM:\s*flow\](?:\*\*)?", stripped, re.IGNORECASE):
                flush_para()
                flow_next = True
                index += 1
                continue

            if re.fullmatch(r"(?:\*\*)?\[DIAGRAM:\s*lanes\](?:\*\*)?", stripped, re.IGNORECASE):
                flush_para()
                lanes_next = True
                index += 1
                continue

            # "Flow: A1>A2>B2; C1>C2" gives the arrows of the lanes diagram just above it.
            if stripped.startswith("Flow:") and slide.blocks and slide.blocks[-1].kind == "lanes":
                flush_para()
                slide.blocks[-1].text = stripped[len("Flow:"):].strip()
                index += 1
                continue

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
                    kind = "flow" if flow_next else "lanes" if lanes_next else "table"
                    slide.blocks.append(Block(kind, rows=rows))
                    flow_next = lanes_next = False
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
                # A numbered list continued from an earlier slide keeps its numbering.
                start = int(re.match(r"\d+", stripped).group()) if number else 1
                slide.blocks.append(Block(kind, items=items, start=start))
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
        # Several note sections can share a slide number where slides were merged.
        for number in numbers:
            out[number] = (out[number] + "\n\n" + text) if number in out else text
    return out


# Layout constants
TITLE_TOP, TITLE_H = Inches(0.50), Inches(1.00)
CONTENT_TOP = Inches(1.65)
CONTENT_BOTTOM = Inches(6.85)
CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP
GUTTER = Inches(0.30)
BLOCK_GAP = Inches(0.12)
TEXT_SHARE = 0.55

FLOW_H = 2.80                              # inches: the three-box diagram on the overview slide
LANES_H = 4.20                             # inches: the swim-lane diagram on the architecture slide
BODY_SIZES = (20, 18, 16, 14, 13, 12)      # the reference deck sets body text at 20-24pt
TABLE_SIZES = (14, 13, 12, 11, 10)
CODE_PT = 12


def _content_width():
    return SLIDE_W - 2 * MARGIN


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Emu(0)
    frame.margin_top = frame.margin_bottom = Emu(0)
    return frame


def _set_bullet(paragraph, *, number_from: int | None = None) -> None:
    """Give a text-box paragraph a hanging bullet like the template's own placeholders."""
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.32)))
    pPr.set("indent", str(-Inches(0.32)))
    if number_from is not None:
        pPr.append(pPr.makeelement(qn("a:buAutoNum"),
                                   {"type": "arabicPeriod", "startAt": str(number_from)}))
    else:
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "◦"}))


def _para(frame, text: str, size: int | float, *, bold=False, italic=False,
          space_after=6, first=False):
    """Add a paragraph in the theme font, so the template decides typeface and colour."""
    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
    paragraph.text = text
    paragraph.space_after = Pt(space_after)
    paragraph.line_spacing = 1.1
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
    return paragraph


def _lines_needed(text: str, width_in: float, size: int | float) -> int:
    chars_per_line = max(12, int((width_in - 0.32) * 72 / (size * 0.50)))
    return max(1, -(-len(text) // chars_per_line))


def _block_height(block, width_in: float, body: int, table_font: int | float) -> float:
    line = body * 1.25 / 72
    if block.kind == "table":
        n_cols = len(block.rows[0])
        widths = [width_in * (0.30 if n_cols >= 4 else 0.36)] if n_cols > 2 else [width_in / n_cols]
        widths += [(width_in - widths[0]) / (n_cols - 1)] * (n_cols - 1) if n_cols > 1 else []
        cell_line = table_font * 1.15 / 72
        total = 0.0
        for row in block.rows:
            lines = 1
            for c, cell in enumerate(row[:n_cols]):
                usable = max(0.4, widths[min(c, len(widths) - 1)] - 0.2)
                per_line = max(6, int(usable * 72 / (table_font * 0.44)))
                lines = max(lines, -(-len(cell) // per_line))
            total += max(0.29, lines * cell_line + 0.08)
        return total + 0.12
    if block.kind == "code":
        return CODE_PT * 1.3 / 72 * len(block.text.split(NEWLINE)) + 0.16
    if block.kind == "flow":
        return FLOW_H + 0.20
    if block.kind == "lanes":
        return LANES_H + 0.15
    if block.kind in ("bullets", "numbers"):
        return sum(line * _lines_needed(i, width_in, body) + body * 0.6 / 72
                   for i in block.items) + 0.05
    if block.kind == "heading":
        return line * 1.3 + 0.08
    return line * _lines_needed(block.text, width_in, body) + 0.10


def _estimate_height(blocks, width_in: float, body: int, table_font: int | float) -> float:
    return sum(_block_height(b, width_in, body, table_font)
               for b in blocks if b.kind != "figure")


def _pick_sizes(blocks, width_in: float) -> tuple[int, float]:
    budget = CONTENT_H / Inches(1)
    for body in BODY_SIZES:
        for table_font in TABLE_SIZES:
            if _estimate_height(blocks, width_in, body, table_font) <= budget:
                return body, table_font
    return BODY_SIZES[-1], TABLE_SIZES[-1]


def _add_table(slide, rows, left, top, width, font_size: int | float):
    """A table in the template's default table style, as the reference deck uses."""
    n_rows, n_cols = len(rows), len(rows[0])
    row_h = Inches(0.30)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, row_h * n_rows)
    table = shape.table

    if n_cols > 1:
        weights = []
        for c in range(n_cols):
            longest = max((len(row[c]) for row in rows if c < len(row)), default=4)
            average = sum(len(row[c]) for row in rows if c < len(row)) / max(1, len(rows))
            weights.append(max(4.0, (longest + 2 * average) / 3))
        total = sum(weights)
        floor = 0.7 * 914400
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
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.space_after = Pt(0)
            for run in paragraph.runs:
                run.font.size = Pt(font_size)
    height = Inches(_block_height(Block("table", rows=rows), width / Inches(1), BODY_SIZES[0], font_size))
    return shape, height


def _draw_flow(slide, rows, left, top, width) -> None:
    """Draw a table as connected boxes: one column per box, left to right, joined by arrows.

    The header row names each box; every later row is one line inside it. This is the
    overview slide's picture of the project, drawn from CONTENT like every other word on
    the deck, so the diagram and the slide text cannot disagree.
    """
    n = len(rows[0])
    arrow_w, gap = Inches(0.50), Inches(0.08)
    box_w = int((width - (n - 1) * (arrow_w + 2 * gap)) / n)
    head_h, body_h = Inches(0.62), Inches(FLOW_H) - Inches(0.62)
    for c in range(n):
        x = int(left + c * (box_w + arrow_w + 2 * gap))
        head = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, top, box_w, head_h)
        head.fill.solid()
        head.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1
        head.line.fill.background()
        frame = head.text_frame
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        _para(frame, rows[0][c], 22, bold=True, space_after=0, first=True)
        frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        for run in frame.paragraphs[0].runs:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        body = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, top + head_h, box_w, body_h)
        body.fill.solid()
        body.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        body.line.color.theme_color = MSO_THEME_COLOR.ACCENT_1
        body.line.width = Pt(1.5)
        frame = body.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.TOP
        frame.margin_left = frame.margin_right = Inches(0.12)
        frame.margin_top = Inches(0.10)
        lines = [row[c] for row in rows[1:] if c < len(row) and row[c]]
        for i, text in enumerate(lines):
            last = i == len(lines) - 1
            para = _para(frame, text, 14 if last else 16, italic=last, bold=(i == 0),
                         space_after=6, first=(i == 0))
            para.alignment = PP_ALIGN.CENTER
            for run in para.runs:
                run.font.color.theme_color = MSO_THEME_COLOR.TEXT_1

        if c < n - 1:
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, x + box_w + gap,
                int(top + (head_h + body_h) / 2 - Inches(0.30)), arrow_w, Inches(0.60))
            arrow.fill.solid()
            arrow.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1
            arrow.line.fill.background()


def _arrow(slide, x1, y1, x2, y2, *, dashed=False) -> None:
    """A straight connector with an arrowhead at (x2, y2)."""
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), int(x2), int(y2))
    line.line.width = Pt(2.5)
    line.line.color.theme_color = MSO_THEME_COLOR.TEXT_1
    if dashed:
        line.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    ln = line.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def _draw_lanes(slide, rows, flow, left, top, width) -> None:
    """Draw a table as swim lanes: one band per row, one box per non-empty cell.

    Layout comes from the table itself, so the picture is edited where the words are. The header
    row is ignored. In every later row the first cell names the lane and each further cell is a
    grid position: an empty cell leaves a gap, so a box can sit directly under the box that feeds
    it. Text before " — " is the box's title, text after it the detail. A cell starting with "!"
    is drawn filled, for the outputs a reader should find first.

    Arrows come from the "Flow:" line under the table: cells are named by lane letter and column
    number (A1 is the first box of the first lane), chained with ">", chains separated by ";".
    The FIRST chain is the main path, and its boxes carry step numbers 1, 2, 3 in that order, so
    the reading order survives a lane that runs right to left. A box that appears only in later
    chains is a side input -- here, data used once offline -- and is drawn dashed, with a dashed
    arrow and no number, so it cannot be mistaken for a step every pond goes through.
    """
    chains = [[n.strip() for n in c.split(">")] for c in flow.split(";") if c.strip()]
    main = chains[0] if chains else []
    side = {n for chain in chains[1:] for n in chain} - set(main)

    lanes = rows[1:]
    n_lanes, n_cols = len(lanes), max(len(r) for r in lanes) - 1
    label_w, col_gap, lane_gap = Inches(1.95), Inches(0.55), Inches(0.30)
    grid_left = left + label_w
    box_w = int((width - label_w - (n_cols - 1) * col_gap) / n_cols)
    lane_h = int((Inches(LANES_H) - (n_lanes - 1) * lane_gap) / n_lanes)
    pad = Inches(0.10)
    boxes = {}

    for r, lane in enumerate(lanes):
        y = int(top + r * (lane_h + lane_gap))
        band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, y, width, lane_h)
        band.fill.solid()
        band.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1
        band.fill.fore_color.brightness = 0.75
        band.line.fill.background()
        label = _textbox(slide, left + Inches(0.12), y, label_w - Inches(0.2), lane_h)
        label.vertical_anchor = MSO_ANCHOR.MIDDLE
        _para(label, lane[0], 18, bold=True, space_after=0, first=True)

        for c, cell in enumerate(lane[1:]):
            if not cell.strip():
                continue
            name = f"{chr(ord('A') + r)}{c + 1}"
            filled = cell.startswith("!")
            title, _, detail = cell.lstrip("!").strip().partition(" — ")
            x = int(grid_left + c * (box_w + col_gap))
            box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y + pad, box_w,
                                         lane_h - 2 * pad)
            box.adjustments[0] = 0.12
            box.fill.solid()
            if filled:
                box.fill.fore_color.theme_color = MSO_THEME_COLOR.ACCENT_1
                box.fill.fore_color.brightness = -0.25
            else:
                box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            box.line.color.theme_color = MSO_THEME_COLOR.ACCENT_1
            box.line.width = Pt(1.5)
            if name in side:
                box.line.color.theme_color = MSO_THEME_COLOR.TEXT_1
                box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            frame = box.text_frame
            frame.word_wrap = True
            frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            frame.margin_left = frame.margin_right = Inches(0.10)
            frame.margin_top = frame.margin_bottom = Inches(0.04)
            parts = [(title, 16, True)] + ([(detail, 13, False)] if detail else [])
            for i, (text, size, bold) in enumerate(parts):
                para = _para(frame, text, size, bold=bold, space_after=2, first=(i == 0))
                para.alignment = PP_ALIGN.CENTER
                for run in para.runs:
                    if filled:
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    else:
                        run.font.color.theme_color = MSO_THEME_COLOR.TEXT_1
            boxes[name] = (x, y + pad, box_w, lane_h - 2 * pad)

    for chain_index, names in enumerate(chains):
        for a, b in zip(names, names[1:]):
            ax, ay, aw, ah = boxes[a]
            bx, by, bw, bh = boxes[b]
            dashed = chain_index > 0
            if a[0] == b[0]:                       # same lane: left or right
                y = ay + ah / 2
                if bx > ax:
                    _arrow(slide, ax + aw, y, bx, y, dashed=dashed)
                else:
                    _arrow(slide, ax, y, bx + bw, y, dashed=dashed)
            else:                                  # next lane: straight down
                x = ax + aw / 2
                _arrow(slide, x, ay + ah, x, by, dashed=dashed)

    # Step numbers last, so they sit on top of the boxes and arrows they straddle.
    badge = Inches(0.36)
    for step, name in enumerate(main, start=1):
        x, y, _, _ = boxes[name]
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, int(x - badge / 2), int(y - badge / 2),
                                     badge, badge)
        dot.fill.solid()
        dot.fill.fore_color.theme_color = MSO_THEME_COLOR.TEXT_1
        dot.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        dot.line.width = Pt(1.25)
        frame = dot.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = _para(frame, str(step), 13, bold=True, space_after=0, first=True)
        para.alignment = PP_ALIGN.CENTER
        for run in para.runs:
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


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


def _replace_runs(paragraph, texts: list[str]) -> None:
    """Overwrite a paragraph's runs in place, keeping each run's formatting."""
    runs = paragraph.runs
    for run, text in zip(runs, texts):
        run.text = text
    for run in runs[len(texts):]:
        run._r.getparent().remove(run._r)


def _open_template():
    """Load the template and keep only its title slide and its closing slide."""
    prs = Presentation(TEMPLATE)
    id_list = prs.slides._sldIdLst
    entries = list(id_list)
    for index, entry in enumerate(entries):
        if index not in (0, len(entries) - 1):
            prs.part.drop_rel(entry.rId)
            id_list.remove(entry)
    # New slides are named slide<N>.xml by count, which would collide with the kept
    # closing slide's original name.
    prs.slides[1].part.partname = PackURI("/ppt/slides/slide999.xml")
    props = prs.core_properties
    props.author = props.last_modified_by = "G Mohan Jayanth"
    props.title = "Review 3 - 19MIS1127"
    props.subject = props.keywords = props.comments = ""
    return prs


def fill_title_slide(slide, slide_data) -> None:
    """The template's title slide, with the registered title and this project's details."""
    registered = ("An Edge-Deployable Multimodal Deep Learning Framework for Early Shrimp "
                  "Disease Detection Using Image")
    heading = next((b.text for b in slide_data.blocks
                    if "shrimp" in b.text.lower() and len(b.text) > 30), registered)
    details = [("Registration Number: ", "19MIS1127"),
               ("Name:", " G Mohan Jayanth"),
               ("Guide Name:", " Dr. M. Premalatha")]
    tagline = next((b.text.split(":", 1)[1].strip() for b in slide_data.blocks
                    if b.kind == "para" and b.text.startswith("Tagline:")), "")
    for shape in slide.placeholders:
        idx = shape.placeholder_format.idx
        if idx == 0:
            _replace_runs(shape.text_frame.paragraphs[0], [heading.upper()])
            if tagline:
                # One plain line under the registered title saying what the project does.
                # The title itself stays verbatim; the tagline only adds to it.
                frame = _textbox(slide, shape.left, shape.top + shape.height + Inches(0.05),
                                 shape.width, Inches(0.45))
                _para(frame, tagline, 20, italic=True, space_after=0, first=True)
                frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        elif idx == 1:
            for paragraph, pair in zip(shape.text_frame.paragraphs, details):
                _replace_runs(paragraph, list(pair))
    if slide_data.notes:
        slide.notes_slide.notes_text_frame.text = _clean_notes(slide_data.notes)


def fill_closing_slide(prs, slide, slide_data) -> None:
    """Move the template's closing slide to the current end of the deck and set its text."""
    id_list = prs.slides._sldIdLst
    for entry in list(id_list):
        if prs.part.related_part(entry.rId) is slide.part:
            id_list.remove(entry)
            id_list.append(entry)
    text = next((b.text for b in slide_data.blocks if b.kind == "para"), slide_data.title)
    _replace_runs(slide.shapes.title.text_frame.paragraphs[0], [text.upper()])
    if slide_data.notes:
        slide.notes_slide.notes_text_frame.text = _clean_notes(slide_data.notes)


def _clean_notes(text: str) -> str:
    text = re.sub(r"(?m)^#+\s*", "", text)               # markdown headings read as plain lines
    return _strip_md(_rejoin_wrapped_emphasis(text)).replace("**", "").replace("`", "")


def render_slide(prs, slide_data) -> None:
    layout = next(l for l in prs.slide_layouts if l.name == TITLE_AND_CONTENT)
    slide = prs.slides.add_slide(layout)

    title = slide.shapes.title
    title.left, title.top, title.width, title.height = MARGIN, TITLE_TOP, _content_width(), TITLE_H
    title.text = slide_data.title
    title.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    for run in title.text_frame.paragraphs[0].runs:
        run.font.size = Pt(40 if len(slide_data.title) <= 30 else 32)
    # Content is placed as shapes below, so the body placeholder is not needed.
    for shape in list(slide.placeholders):
        if shape.placeholder_format.idx != 0:
            shape._element.getparent().remove(shape._element)

    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    has_figure = any((PROJECT_ROOT / f.path).exists() for f in figures)

    if has_figure:
        text_w = Emu(int(_content_width() * TEXT_SHARE))
        fig_left = MARGIN + text_w + GUTTER
        fig_w = _content_width() - text_w - GUTTER
        _place_figures(slide, figures, fig_left, CONTENT_TOP, fig_w, CONTENT_H)
        render_w = text_w
    else:
        render_w = _content_width()

    width_in = render_w / Inches(1)
    body_pt, table_pt = _pick_sizes(others, width_in)

    top = CONTENT_TOP
    for block in others:
        if block.kind == "table" and block.rows:
            _, drawn_h = _add_table(slide, block.rows, MARGIN, top, render_w, table_pt)
            top += drawn_h + BLOCK_GAP
        elif block.kind == "lanes" and block.rows:
            _draw_lanes(slide, block.rows, block.text, MARGIN, top, render_w)
            top += Inches(_block_height(block, width_in, body_pt, table_pt))
        elif block.kind == "flow" and block.rows:
            _draw_flow(slide, block.rows, MARGIN, top, render_w)
            top += Inches(_block_height(block, width_in, body_pt, table_pt))
        elif block.kind == "code":
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            for i, line in enumerate(block.text.split(NEWLINE)):
                p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
                p.text = line
                p.space_after = Pt(0)
                p.line_spacing = 1.0
                for run in p.runs:
                    run.font.name = "Consolas"
                    run.font.size = Pt(CODE_PT)
            top += h + BLOCK_GAP
        elif block.kind == "heading":
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            _para(frame, block.text, body_pt + 2, bold=True, space_after=2, first=True)
            top += h
        elif block.kind in ("bullets", "numbers"):
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            for i, item in enumerate(block.items):
                paragraph = _para(frame, item, body_pt, space_after=body_pt * 0.45, first=(i == 0))
                _set_bullet(paragraph, number_from=block.start if block.kind == "numbers" else None)
            top += h + BLOCK_GAP
        elif block.kind in ("para", "quote"):
            h = Inches(_block_height(block, width_in, body_pt, table_pt))
            frame = _textbox(slide, MARGIN, top, render_w, h)
            _para(frame, block.text, body_pt, italic=(block.kind == "quote"), first=True)
            top += h + BLOCK_GAP

    # The source file behind each slide goes in the notes, not on the slide face.
    notes = _clean_notes(slide_data.notes) if slide_data.notes else ""
    if slide_data.source:
        notes = (notes + "\n\n" if notes else "") + f"Source: {slide_data.source}"
    if notes:
        slide.notes_slide.notes_text_frame.text = notes


def _paginate(slide_data: Slide) -> list[Slide]:
    """Check if content fits on a single slide; if not, split gracefully."""
    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    has_figure = any((PROJECT_ROOT / f.path).exists() for f in figures)

    render_w = _content_width() * TEXT_SHARE if has_figure else _content_width()
    width_in = render_w / Inches(1)

    budget = CONTENT_H / Inches(1)
    body_pt, table_pt = _pick_sizes(others, width_in)

    total_h = _estimate_height(others, width_in, body_pt, table_pt)
    if total_h <= budget or len(others) <= 1:
        return [slide_data]

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

    prs = _open_template()
    title_slide, closing_slide = prs.slides[0], prs.slides[1]

    split = []
    for slide in slides:
        if slide.number == "1":
            fill_title_slide(title_slide, slide)
            continue
        if slide.title.lower() == "thank you":
            fill_closing_slide(prs, closing_slide, slide)
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
