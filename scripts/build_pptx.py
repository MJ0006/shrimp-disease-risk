"""Build `Review2_19MIS1127.pptx` from the slide content file.

Why this exists, given the rule against it
------------------------------------------
The Master Prompt is explicit: *"The actual PowerPoint presentation should NOT be created unless
explicitly requested"* (Section 26, non-negotiable #35). The student requested it directly on 2026-08-28,
which is the condition the rule names. `D-032` records the request and this script's provenance.

Why generated rather than typed
-------------------------------
Same standing rule the figures (`D-014`), the demonstration runner (`D-022`) and the notebook
(`D-023`) follow: **any artefact that presents results is generated from the artefacts it presents,
and is regenerated when they change.** A hand-built deck goes stale the moment a table is
regenerated, and a stale deck is worse than none because it looks authoritative in a room where
nobody can check it.

So the deck is a *rendering* of `docs/presentation/ppt_content.md`, which is itself verified against
`reports/`. Nothing is authored here. If a number is wrong on a slide, it is wrong in the content
file, and that is where it gets fixed.

What goes where
---------------
* **On the slide** — headings, bullets, tables, figures, and a small grey source line, because a
  panel that can see where a number came from asks a different class of question.
* **In the speaker notes** — the matching section of `speaker_notes.md`, plus any block that is
  guidance to the presenter rather than content for the audience ("Read this before the review",
  "say this out loud"). Those would be baffling on a projected slide and are exactly what the notes
  pane is for.

Overflow
--------
Slide content is dense and PowerPoint will not warn you when text runs off the bottom. The script
estimates the rendered height of every slide and prints a warning for any that is likely to
overflow, so the student can trim deliberately rather than discovering it on the projector.

Usage
-----
    .venv/Scripts/python.exe scripts/build_pptx.py
    .venv/Scripts/python.exe scripts/build_pptx.py --out "Review2.pptx"
    .venv/Scripts/python.exe scripts/build_pptx.py --no-appendix   # drop the optional A1 slide
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
CONTENT = PROJECT_ROOT / "docs" / "presentation" / "ppt_content.md"
NOTES = PROJECT_ROOT / "docs" / "presentation" / "speaker_notes.md"

# 16:9. Every projector in a review room is widescreen; 4:3 letterboxes and wastes the width the
# result tables need.
SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)

INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x0B, 0x4F, 0x8A)      # heading blue
MUTED = RGBColor(0x5A, 0x5A, 0x5A)       # source lines
RULE = RGBColor(0xD8, 0xDD, 0xE2)
HEADER_FILL = RGBColor(0xEC, 0xF2, 0xF8)

MARGIN = Inches(0.55)
TITLE_H = Inches(0.95)

# Blocks that address the presenter, not the audience. They belong in the notes pane; projected,
# they read as instructions the room was not meant to see.
PRESENTER_MARKERS = (
    "read this table before the review", "read this before the review", "say this out loud",
    "do not \"correct\" it", "be honest about", "if challenged", "never say",
    "what to say", "do not read", "verification note",
)


# ---------------------------------------------------------------------------------------
# Parsing the content file
# ---------------------------------------------------------------------------------------

@dataclass
class Block:
    kind: str                      # heading | para | bullets | numbers | table | figure | quote
    text: str = ""
    items: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    path: str = ""
    start: int = 1          # first number, so a split numbered list keeps counting


@dataclass
class Slide:
    number: str
    title: str
    blocks: list[Block] = field(default_factory=list)
    source: str = ""
    notes: str = ""


def _strip_md(text: str) -> str:
    """Markdown to plain text. Emphasis is carried by layout here, not by asterisks on a slide."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"~~([^~]*)~~", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.replace("&nbsp;", " ").strip()


def _is_presenter_note(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in PRESENTER_MARKERS)


def parse_content(markdown: str) -> list[Slide]:
    parts = re.split(r"^## (Slide [^\n]*)$", markdown, flags=re.M)
    slides: list[Slide] = []

    for i in range(1, len(parts), 2):
        heading, body = parts[i].strip(), parts[i + 1]

        # "Slide budget and drop order" is guidance for the student, not a slide.
        # "Slide A1 (optional appendix) — ..." carries a parenthetical before the dash.
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
                # A fenced block is a diagram or an equation chain. Its line breaks ARE the
                # content -- flattening it into a paragraph destroys the architecture slide.
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

            # The content file writes both "[FIGURE: x]" and "[FIGURES: x . y]". Only the
            # singular was matched, so slide 20 announced two figures and showed neither.
            figure = re.match(r"\*\*\[(FIGURES?|SCREENSHOT):\s*([^\]]+)\]\*\*", stripped)
            if figure:
                flush_para()
                for candidate in re.split(r"\s*[·,]\s*|\s+and\s+", figure.group(2)):
                    candidate = candidate.strip().strip("`")
                    if not candidate:
                        continue
                    if "/" not in candidate:            # bare filename -> the figures directory
                        candidate = "reports/figures/" + candidate
                    slide.blocks.append(Block("figure", path=candidate))
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
                if text.startswith("#"):                       # the title block on slide 1
                    slide.blocks.append(Block("heading", _strip_md(text.lstrip("# "))))
                elif _is_presenter_note(text):
                    note_lines.append(text)
                elif text:
                    slide.blocks.append(Block("quote", _strip_md(text)))
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
                        # A wrapped continuation of the previous item. This has to test the RAW
                        # line: stripping first destroys the indentation that identifies it, which
                        # broke every entry on the References slide into its own list and restarted
                        # the numbering at 1 each time.
                        items[-1] += ' ' + _strip_md(current)
                        index += 1
                    else:
                        break
                # Emphasis that spans a wrapped line leaves an unmatched ** on each half;
                # stripping per line cannot see the pair, so strip once more on the join.
                items = [_strip_md(item) for item in items]
                slide.blocks.append(Block(kind, items=items))
                continue

            pending.append(stripped)
            index += 1

        flush_para()
        # A slide that already carries a table or a figure has made its point visually.
        # Long paragraphs beside it are commentary, and belong where the presenter reads them.
        has_visual = any(b.kind in ("table", "figure") for b in slide.blocks)
        if has_visual:
            kept = []
            for block in slide.blocks:
                if block.kind in ("para", "quote") and len(block.text) > PROSE_TO_NOTES:
                    note_lines.append(block.text)
                else:
                    kept.append(block)
            slide.blocks = kept
        slide.notes = _drop_anticipated_questions(NEWLINE.join(note_lines))
        slides.append(slide)

    return slides


def parse_notes(markdown: str) -> dict[str, str]:
    """Speaker notes keyed by slide number. Grouped sections ("Slides 6-8") map to every slide
    they cover, so no slide reaches the deck without notes attached."""
    out: dict[str, str] = {}
    parts = re.split(r"^# (Slide[^\n]*)$", markdown, flags=re.M)
    for i in range(1, len(parts), 2):
        heading, body = parts[i], parts[i + 1]
        numbers = re.findall(r"A?\d+", heading)
        if not numbers:
            continue
        if "–" in heading or "-" in heading.replace("—", ""):
            span = re.search(r"(\d+)\s*[–-]\s*(\d+)", heading)
            if span:
                numbers = [str(n) for n in range(int(span.group(1)), int(span.group(2)) + 1)]
        text = re.sub(r"\n{3,}", "\n\n", _strip_md_block(body)).strip()
        text = _drop_anticipated_questions(text)
        for number in numbers:
            out.setdefault(number, text)
    return out


# Headings whose content is a question attributed to the panel, plus the prepared reply.
# They stay in `speaker_notes.md` -- that is what the student revises from, and Master
# Prompt Section 27 is satisfied there. They are cut from the .pptx because the deck gets
# handed over: a panel member opening the notes pane would find questions written in their
# mouth, some adversarially framed. Easy to avoid, and a bad first impression if not.
QUESTION_HEADINGS = ("Possible Questions", "Defensible Answers", "Possible Question",
                     "Defensible Answer", "Questions you will be asked",
                     "Things to admit immediately if asked")


def _drop_anticipated_questions(text: str) -> str:
    """Remove the anticipated-question blocks from notes bound for the .pptx."""
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
    """Join lines that split a **bold** span, so the markers can be stripped as a pair."""
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


# ---------------------------------------------------------------------------------------
# Rendering
#
# The layout is a fixed grid rather than a stack of guessed offsets. Everything below places
# shapes against the same four numbers -- CONTENT_TOP, CONTENT_BOTTOM, MARGIN and GUTTER -- so
# left edges line up down the deck and no block can be positioned relative to a mis-measured
# neighbour. The first version of this file stacked blocks by estimated height, which drifted:
# two slides ended up with overlapping shapes and one ran 0.8 inch off the bottom edge.
# ---------------------------------------------------------------------------------------

CONTENT_TOP = Inches(1.32)
CONTENT_BOTTOM = Inches(6.92)
CONTENT_H = CONTENT_BOTTOM - CONTENT_TOP
GUTTER = Inches(0.30)
BLOCK_GAP = Inches(0.13)
TEXT_SHARE = 0.55

# The deck-wide type scale. 15pt body is readable from the back of a room; 11pt tables
# are the smallest that stay legible when projected.
BODY_PT, TABLE_PT = 15, 11
# Prose longer than this, on a slide that already carries a table or a figure, is
# commentary rather than content: the visual makes the point and the paragraph explains
# it. It moves to the notes pane, where the presenter reads it and the room does not.
PROSE_TO_NOTES = 120

BULLET_RE = r"[-*]\s+(.*)"
NUMBER_RE = r"\d+\.\s+(.*)"          # width given to text when a figure shares the slide


def _content_width():
    return SLIDE_W - 2 * MARGIN


def _textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Emu(0)
    frame.margin_top = frame.margin_bottom = Emu(0)
    return frame


def _para(frame, text: str, size: int, *, bold=False, italic=False, colour=INK,
          space_after=6, bullet=False, first=False):
    paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
    paragraph.text = ("•  " + text) if bullet else text
    paragraph.space_after = Pt(space_after)
    paragraph.line_spacing = 1.18
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = colour
        run.font.name = "Calibri"
    return paragraph


# --- measurement -----------------------------------------------------------------------
# Deliberately conservative. Under-estimating puts one block on top of the next; over-estimating
# only leaves white space, so every figure here rounds up.

def _lines_needed(text: str, width_in: float, size: int) -> int:
    chars_per_line = max(12, int(width_in * 72 / (size * 0.48)))
    return max(1, -(-len(text) // chars_per_line))


def _block_height(block, width_in: float, body: int, table_font: int) -> float:
    """Height in inches, rounded up."""
    line = body * 1.24 / 72
    if block.kind == "table":
        # PowerPoint treats a row height as a minimum and grows the row to fit its content, so
        # the estimate has to model the wrapping rather than assume one line per row. Getting
        # this wrong in either direction is visible: too low overlaps the next block, too high
        # paginates a slide that would have fitted.
        n_cols = len(block.rows[0])
        widths = [width_in * (0.30 if n_cols >= 4 else 0.36)] if n_cols > 2 else [width_in / n_cols]
        widths += [(width_in - widths[0]) / (n_cols - 1)] * (n_cols - 1) if n_cols > 1 else []
        cell_line = table_font * 1.14 / 72
        total = 0.0
        for row in block.rows:
            lines = 1
            for c, cell in enumerate(row[:n_cols]):
                usable = max(0.4, widths[min(c, len(widths) - 1)] - 0.14)
                per_line = max(6, int(usable * 72 / (table_font * 0.48)))
                lines = max(lines, -(-len(cell) // per_line))
            total += max(0.24, lines * cell_line + 0.06)
        return total + 0.14
    if block.kind == "code":
        return 0.022 * 10 * len(block.text.split(NEWLINE)) + 0.18
    if block.kind in ("bullets", "numbers"):
        return sum(line * _lines_needed(i, width_in, body) + 0.030 for i in block.items) + 0.06
    return line * _lines_needed(block.text, width_in, body) + 0.10


def _estimate_height(blocks, width_in: float, body: int, table_font: int) -> float:
    return sum(_block_height(b, width_in, body, table_font)
               for b in blocks if b.kind != "figure")


def _is_reference_list(title: str) -> bool:
    """The references slide sets its list at the table size rather than the body size.

    Ten citations at 15pt ran to three slides. 11pt is not a new size in this deck -- every
    table already uses it -- and a citation list is reference material of exactly that kind:
    nobody reads it from the back of the room, they read it in the handout.
    """
    return title.lower().startswith("references")


def _pick_sizes(blocks, width_in: float) -> tuple[int, int]:
    """One type scale for the entire deck.

    An earlier version chose a size per slide, from 18pt down to 11pt, so that everything fitted.
    It fitted, and it looked like eight different decks: the reader notices the type changing from
    slide to slide long before they notice why. Consistency is worth more than density, so the
    size is fixed and anything that does not fit is moved to the notes or continued onto a second
    slide -- both of which are visible decisions rather than silent ones.
    """
    # Body text is fixed: it is what a reader notices changing from slide to slide. A dense
    # reference table may drop a step or two, because splitting a nine-paper literature table
    # across two slides leaves the second one nearly empty -- which is exactly what it looked
    # like. 9pt is the floor; below that a projected table stops being readable and the slide
    # should be split instead.
    budget = CONTENT_H / Inches(1) * 1.06
    for table_font in (TABLE_PT, 10, 9):
        if _estimate_height(blocks, width_in, BODY_PT, table_font) <= budget:
            return BODY_PT, table_font
    return BODY_PT, 9


# --- components ------------------------------------------------------------------------

def _title_bar(slide, text: str, number: str) -> None:
    bar = slide.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, Inches(1.02))
    bar.fill.solid()
    bar.fill.fore_color.rgb = HEADER_FILL
    bar.line.fill.background()
    bar.shadow.inherit = False

    accent = slide.shapes.add_shape(1, Emu(0), Inches(1.02), SLIDE_W, Inches(0.035))
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT
    accent.line.fill.background()
    accent.shadow.inherit = False

    frame = _textbox(slide, MARGIN, Inches(0.16), _content_width() - Inches(1.0), Inches(0.72))
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    size = 25 if len(text) < 58 else (21 if len(text) < 88 else 18)
    _para(frame, text, size, bold=True, colour=ACCENT, space_after=0, first=True)

    tag = _textbox(slide, SLIDE_W - MARGIN - Inches(0.9), Inches(0.16), Inches(0.9), Inches(0.72))
    tag.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = _para(tag, number, 12, bold=True, colour=MUTED, space_after=0, first=True)
    paragraph.alignment = PP_ALIGN.RIGHT


def _add_table(slide, rows, left, top, width, font_size: int):
    n_rows, n_cols = len(rows), len(rows[0])
    row_h = Inches(0.28 if font_size > 10 else 0.25)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, row_h * n_rows)
    table = shape.table

    # Column widths follow the content. A fixed share for the first column gave 'S.No' the
    # same 30% as 'Gap Identified', so the literature tables wasted a third of the slide on
    # two-character numbers while the prose columns wrapped.
    if n_cols > 1:
        weights = []
        for c in range(n_cols):
            longest = max((len(row[c]) for row in rows if c < len(row)), default=4)
            average = sum(len(row[c]) for row in rows if c < len(row)) / max(1, len(rows))
            weights.append(max(4.0, (longest + 2 * average) / 3))
        total = sum(weights)
        floor = 0.55 * 914400                      # no column narrower than 0.55in
        widths = [max(floor, width * w / total) for w in weights]
        overflow = sum(widths) - width
        if overflow > 0:                            # take the excess from the widest columns
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
    """Fit each figure inside its slot on BOTH axes, preserving aspect ratio.

    Scaling on width alone is what pushed slide 16 off the bottom of the page: two stacked
    figures, each as wide as the column, came to more than the column was tall.
    """
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
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    band = slide.shapes.add_shape(1, Emu(0), Emu(0), SLIDE_W, Inches(0.3))
    band.fill.solid()
    band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    band.shadow.inherit = False

    heading = next((b.text for b in slide_data.blocks if b.kind == "heading"), slide_data.title)
    frame = _textbox(slide, MARGIN + Inches(0.35), Inches(1.7), _content_width() - Inches(0.7),
                     Inches(2.3))
    # Large type needs more leading than body text, and the title is the one place a
    # collision is unmissable: at 100% the descenders of line 1 met the ascenders of line 2.
    paragraph = _para(frame, heading, 30, bold=True, colour=ACCENT, space_after=0, first=True)
    paragraph.line_spacing = 1.28

    rule = slide.shapes.add_shape(1, MARGIN + Inches(0.35), Inches(4.25),
                                  _content_width() - Inches(0.7), Emu(9525))
    rule.fill.solid()
    rule.fill.fore_color.rgb = RULE
    rule.line.fill.background()
    rule.shadow.inherit = False

    # Only the four things the format asks for. Anything else crowds a title slide -- the
    # block that used to land here, 'what each word of the title commits you to', wrapped
    # into the footer. It is preparation material, so it moves to the notes rather than
    # being dropped: it is the answer to a panel testing any word of the title.
    wanted = ("Registration Number", "Student Name", "Guide", "Reg")
    details = [b.text for b in slide_data.blocks
               if b.kind == "para" and any(b.text.startswith(w) for w in wanted)][:3]

    spare = []
    for block in slide_data.blocks:
        if block.kind == "table":
            spare.append(" | ".join(block.rows[0]))
            spare.extend("  " + " | ".join(row) for row in block.rows[1:])
        elif block.kind in ("para", "quote") and block.text not in details:
            spare.append(block.text)
    if spare:
        slide_data.notes = (slide_data.notes + NEWLINE + NEWLINE +
                            "From the content file, kept off the slide:" + NEWLINE +
                            NEWLINE.join(spare)).strip()
    frame = _textbox(slide, MARGIN + Inches(0.35), Inches(4.6), _content_width() - Inches(0.7),
                     Inches(1.8))
    for i, line in enumerate(details):
        _para(frame, line, 16, bold=("Guide" in line), space_after=10, first=(i == 0))

    footer = _textbox(slide, MARGIN + Inches(0.35), Inches(6.65), _content_width(), Inches(0.4))
    _para(footer, "Review 2  ·  2026-09-02", 12, colour=MUTED, space_after=0, first=True)


def _clean_notes(text: str) -> str:
    """Last pass over notes text: no stray markdown reaches the notes pane either."""
    return _strip_md(_rejoin_wrapped_emphasis(text)).replace("**", "").replace("`", "")


def render_slide(prs, slide_data) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _title_bar(slide, slide_data.title, slide_data.number)

    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    has_figure = any((PROJECT_ROOT / f.path).exists() for f in figures)

    if has_figure and others:
        text_width = Emu(int(_content_width() * TEXT_SHARE))
        figure_left = MARGIN + text_width + GUTTER
        figure_width = SLIDE_W - MARGIN - figure_left
    else:
        text_width = _content_width()
        figure_left, figure_width = MARGIN, _content_width()

    body, table_font = _pick_sizes(others, text_width / Inches(1))
    if _is_reference_list(slide_data.title):
        body = table_font
    top = CONTENT_TOP

    for position, block in enumerate(others):
        height = Inches(_block_height(block, text_width / Inches(1), body, table_font))
        # Never drop a block. The paginator allows a slide 6% over budget before splitting it,
        # so the renderer has to accept the same overshoot -- when it did not, a table that the
        # paginator had decided to keep whole was silently skipped here, and the third
        # literature slide rendered with no table at all while every test still passed.
        if position and top + height > CONTENT_BOTTOM + Inches(0.4):
            break                                  # pagination handles the remainder
        if block.kind == "table":
            _add_table(slide, block.rows, MARGIN, top, text_width, table_font)
        elif block.kind == "code":
            lines = block.text.split(NEWLINE)
            widest = max(len(line) for line in lines)
            mono = 11
            while mono > 6 and widest * mono * 0.60 > (text_width / Inches(1)) * 72:
                mono -= 1
            frame = _textbox(slide, MARGIN, top, text_width, height)
            for i, line in enumerate(lines):
                paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
                paragraph.text = line or " "
                paragraph.space_after = Pt(0)
                paragraph.line_spacing = 1.05
                for run in paragraph.runs:
                    run.font.size = Pt(mono)
                    run.font.name = "Consolas"
                    run.font.color.rgb = INK
        elif block.kind in ("bullets", "numbers"):
            frame = _textbox(slide, MARGIN, top, text_width, height)
            for i, item in enumerate(block.items):
                prefix = f"{block.start + i}.  " if block.kind == "numbers" else ""
                _para(frame, prefix + item, body, bullet=(block.kind == "bullets"),
                      space_after=4, first=(i == 0))
        elif block.kind == "quote":
            frame = _textbox(slide, MARGIN + Inches(0.14), top,
                             text_width - Inches(0.14), height)
            _para(frame, block.text, body, italic=True, colour=ACCENT, space_after=0, first=True)
        else:
            frame = _textbox(slide, MARGIN, top, text_width, height)
            _para(frame, block.text, body, bold=(block.kind == "heading"), space_after=0,
                  first=True)
        top = top + height + BLOCK_GAP

    if has_figure:
        _place_figures(slide, figures, figure_left, CONTENT_TOP, figure_width,
                       CONTENT_H if others else CONTENT_H - Inches(0.2))
    elif figures:
        frame = _textbox(slide, MARGIN, top, _content_width(), Inches(0.4))
        _para(frame, f"[figure not found: {figures[0].path}]", 11, italic=True, colour=MUTED,
              space_after=0, first=True)

    if slide_data.source:
        frame = _textbox(slide, MARGIN, Inches(7.0), _content_width(), Inches(0.3))
        _para(frame, "Source: " + slide_data.source, 9, colour=MUTED, space_after=0, first=True)

    if slide_data.notes:
        slide.notes_slide.notes_text_frame.text = _clean_notes(slide_data.notes)


def _row_heights(block, width_in: float, table_font: int) -> list[float]:
    """Rendered height of each table row, header first."""
    n_cols = len(block.rows[0])
    widths = [width_in * (0.30 if n_cols >= 4 else 0.36)] if n_cols > 2 else [width_in / n_cols]
    widths += [(width_in - widths[0]) / (n_cols - 1)] * (n_cols - 1) if n_cols > 1 else []
    cell_line = table_font * 1.14 / 72
    out = []
    for row in block.rows:
        need = 1
        for c, cell in enumerate(row[:n_cols]):
            usable = max(0.4, widths[min(c, len(widths) - 1)] - 0.14)
            per_line = max(6, int(usable * 72 / (table_font * 0.48)))
            need = max(need, -(-len(cell) // per_line))
        out.append(max(0.24, need * cell_line + 0.06))
    return out


def _split_table(block, width_in: float, table_font: int, limit: float) -> list:
    """Deal a table's rows across as many blocks as it takes, repeating the header on each.

    Without this a table taller than one slide is silently dropped: the renderer skips any block
    that will not fit, and the paginator keeps a first block on its page whatever its size. That
    is how both literature tables -- a section the prescribed format requires -- vanished from
    the deck while every test still passed.
    """
    heights = _row_heights(block, width_in, table_font)
    header, header_h = block.rows[0], heights[0]
    if sum(heights) + 0.14 <= limit:
        return [block]

    out, current, used = [], [], header_h + 0.14
    for row, height in zip(block.rows[1:], heights[1:]):
        if current and used + height > limit:
            out.append(Block("table", rows=[header] + current))
            current, used = [], header_h + 0.14
        current.append(row)
        used += height
    if current:
        out.append(Block("table", rows=[header] + current))
    return out


def _split_list(block, width_in: float, body: int, limit: float) -> list:
    """Deal a long bullet or numbered list across pages.

    Only tables could be split, so a Conclusion written as ten numbered points was kept whole by
    the paginator and then drawn past the bottom edge of the slide -- the renderer never drops the
    first block on a page, which is right, but it means an unsplittable block has to be made
    splittable instead.

    Numbered lists keep counting across the break: restarting at 1 on the second slide would
    misrepresent how many conclusions there are.
    """
    line = body * 1.24 / 72
    pages, current, used = [], [], 0.0
    for item in block.items:
        height = line * _lines_needed(item, width_in, body) + 0.030
        if current and used + height > limit:
            pages.append(current)
            current, used = [], 0.0
        current.append(item)
        used += height
    if current:
        pages.append(current)

    out, offset = [], 0
    for page in pages:
        part = Block(block.kind, items=page)
        part.start = offset + 1
        offset += len(page)
        out.append(part)
    return out


def _paginate(slide_data) -> list:
    """Split a slide whose content cannot fit, rather than letting it run off the projector.

    PowerPoint gives no warning when text overflows the slide -- it simply renders past the
    bottom edge, which nobody sees until it is on a screen in front of a panel. Where the
    estimated height exceeds the body, the blocks are dealt onto continuation slides carrying the
    same title with "(cont.)" appended. The alternative, shrinking to 8pt, is unreadable from the
    back of a room, and silently dropping blocks would lose content the student wrote deliberately.
    """
    figures = [b for b in slide_data.blocks if b.kind == "figure"]
    others = [b for b in slide_data.blocks if b.kind != "figure"]
    width_in = (SLIDE_W - 2 * MARGIN) / Inches(1)
    if figures:
        width_in *= 0.54
    body, table_font = _pick_sizes(others, width_in)
    if _is_reference_list(slide_data.title):
        body = table_font
    # 6% tolerance: the estimator rounds up at every step, and splitting a slide that would
    # have fitted is worse than a little white space -- a Conclusion continued onto a second
    # slide reads as a mistake.
    if _estimate_height(others, width_in, body, table_font) <= CONTENT_H / Inches(1) * 1.06:
        return [slide_data]

    limit = CONTENT_H / Inches(1) * 0.94

    # Split any single block that is taller than a page before packing, so nothing can be
    # dropped for being too big to place.
    expanded = []
    for block in others:
        height = _block_height(block, width_in, body, table_font)
        if height > limit and block.kind == "table":
            expanded.extend(_split_table(block, width_in, table_font, limit))
        elif height > limit and block.kind in ("bullets", "numbers"):
            expanded.extend(_split_list(block, width_in, body, limit))
        else:
            expanded.append(block)

    pages, current, used = [], [], 0.0
    for index, block in enumerate(expanded):
        height = _block_height(block, width_in, body, table_font)
        if current and used + height > limit:
            pages.append(current)
            current, used = [], 0.0
        # Keep a short heading-like paragraph with whatever follows it. Left alone, the packer
        # puts the heading at the foot of one page and its table at the top of the next, which
        # reads as a widow and leaves the continuation slide with nothing to be titled after.
        if (block.kind == "para" and len(block.text) < 95 and index + 1 < len(expanded)):
            following = _block_height(expanded[index + 1], width_in, body, table_font)
            if current and used + height + following > limit:
                pages.append(current)
                current, used = [], 0.0
        current.append(block)
        used += height
    if current:
        pages.append(current)

    out = []
    for i, page in enumerate(pages):
        # A continuation that opens with a short line is opening a sub-section -- "B. Assumptions
        # the data cannot let us test". Naming it beats four consecutive slides titled "(cont.)",
        # which tells a reader nothing about where they are.
        title = slide_data.title if i == 0 else f"{slide_data.title} (cont.)"
        if i and page and page[0].kind == "para" and len(page[0].text) < 95:
            # Keep the parent's identity. A slide titled only "Two honest caveats a panel will
            # respect" gives a reader no way to tell which result it belongs to.
            stem = slide_data.title.split(":")[0].split("—")[0].strip()
            title = f"{stem} (cont.) — {page[0].text.rstrip(':')}"
            page = page[1:]
        part = Slide(
            number=slide_data.number if i == 0 else f"{slide_data.number} (cont.)",
            title=title,
            blocks=(figures + page) if i == 0 else page,
            source=slide_data.source if i == len(pages) - 1 else "",
            # A slide with an empty notes pane looks like an oversight during a rehearsal.
            # Continuation slides point back to where their notes actually live.
            notes=(slide_data.notes if i == 0
                   else f"Continued from slide {slide_data.number}. The speaker notes for this "
                        f"material are on that slide."),
        )
        out.append(part)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="Review2_19MIS1127.pptx")
    parser.add_argument("--appendix", action="store_true",
                        help="include the optional 'it runs' appendix slide (a placeholder for a "
                             "screenshot you take on the day; omitted by default because its "
                             "content file entry is presenter guidance, not slide text)")
    args = parser.parse_args(argv)

    slides = parse_content(CONTENT.read_text(encoding="utf-8"))
    notes = parse_notes(NOTES.read_text(encoding="utf-8"))
    if not args.appendix:
        slides = [s for s in slides if not s.number.startswith("A")]

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
            print(f"   slide {number}  ->  '{number}' + '{number} (cont.)'")
        print("   Trim the content file if you would rather keep the original slide count.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
