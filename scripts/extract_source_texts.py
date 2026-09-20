"""Extract plain text from every supplied project input, into the project.

Why this exists
---------------
Stage 0 and Stage 1 audit the supplied inputs claim by claim: "slide 16 says EfficientNet, cell 9
says rexnet_150"; "Table 3 prints t = 2.481 but estimate/SE = 24.7". Those audits were originally
done against text extracted into a *system temp directory*, outside the project. That is a
dependency on something the project does not contain and cannot show a panel -- and it goes away
when the temp directory is cleared.

So the extractions are regenerated here, into `data/interim/extracted_text/`, from the inputs that
live in the project root. Nothing is copied in from outside; everything is rebuilt from a file the
project owns. Re-run this and the evidence behind the audits reappears byte for byte.

    python scripts/extract_source_texts.py

One deliberate exception: identity redaction
--------------------------------------------
The two reference decks carry a third party's name, registration number and guide on slide 1. The
project rule (see CLAUDE.md) is that those decks are prior reference material and must not be
attributed to any named individual in project documents. Those specific lines are therefore replaced
with a visible [REDACTED ...] marker in the extracted text. Nothing else is altered, and the
original `.pptx` files in the project root are untouched -- they remain the primary source.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "interim" / "extracted_text"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

# Lines matching these are personal identity details of a third party. See module docstring.
IDENTITY_PATTERNS = [
    re.compile(r"^\s*Registration Number\s*:", re.I),
    re.compile(r"^\s*Name\s*:", re.I),
    re.compile(r"^\s*Guide\s*Name\s*:", re.I),
]

HEADER = (
    "# Extracted by scripts/extract_source_texts.py from {src}\n"
    "# This is a derived convenience artefact. The original file in the project root is the\n"
    "# primary source and is unmodified.\n"
)


def _redact(line: str) -> str:
    for pat in IDENTITY_PATTERNS:
        if pat.match(line):
            label = line.split(":", 1)[0].strip()
            return f"[REDACTED {label} — third-party identity, see CLAUDE.md]"
    return line


def extract_pdf(src: Path) -> str:
    """Page-delimited text. Page markers matter: every quoted figure cites a page."""
    from pypdf import PdfReader

    reader = PdfReader(str(src))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        parts.append(f"\n\n===== PAGE {i} =====\n{page.extract_text() or ''}")
    return "".join(parts)


def extract_docx(src: Path) -> str:
    """Body walked in document order so paragraph/table sequence is preserved."""
    root = ET.fromstring(zipfile.ZipFile(src).read("word/document.xml"))
    out = []
    for el in root.iter():
        tag = el.tag.split("}")[1]
        if tag == "p":
            txt = "".join(t.text or "" for t in el.iter(W + "t"))
            if txt.strip():
                out.append(txt.strip())
        elif tag == "tr":
            cells = [" ".join(t.text or "" for t in tc.iter(W + "t")).strip()
                     for tc in el.iter(W + "tc")]
            if any(cells):
                out.append(" | ".join(cells))
    return "\n".join(out)


def extract_pptx(src: Path) -> str:
    """Slide text, shape by shape, in slide order -- with identity lines redacted."""
    zf = zipfile.ZipFile(src)
    names = [n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
    names.sort(key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)))

    out = []
    for n in names:
        idx = re.search(r"slide(\d+)\.xml", n).group(1)
        out.append(f"\n----- slide {idx} -----")
        root = ET.fromstring(zf.read(n))
        for shape in root.iter(A + "p"):
            txt = "".join(t.text or "" for t in shape.iter(A + "t")).strip()
            if txt:
                out.append(_redact(txt))
    return "\n".join(out)


def extract_ipynb(src: Path) -> str:
    """Code cells AND stored outputs, kept separate.

    The separation is the point: reported metrics must come from the notebook's own recorded run,
    not from its prose.
    """
    nb = json.loads(src.read_text(encoding="utf-8"))
    out = [f"nbformat {nb.get('nbformat')}  cells {len(nb.get('cells', []))}"]
    for i, cell in enumerate(nb.get("cells", []), 1):
        out.append(f"\n===== CELL {i} ({cell.get('cell_type')}) =====")
        out.append("".join(cell.get("source", [])))
        for o in cell.get("outputs", []):
            text = o.get("text") or o.get("data", {}).get("text/plain") or ""
            if text:
                out.append("--- OUTPUT ---")
                out.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(out)


# (source filename, output name, extractor)
JOBS = [
    ("modeling-shrimp-income-and-disease-risks-prevalence-using-econometric-and-machine-"
     "learning-approaches-evidence-from-vietnam.pdf", "base_paper.txt", extract_pdf),
    ("Review 2 Important Instructions V1.docx", "review2_instructions.txt", extract_docx),
    # The VIT report template, supplied by the student 2026-09-08. Stage 12 writes the report
    # INTO this structure, so its exact section list and front-matter wording must be checkable
    # from text rather than by reopening Word (risk R14, previously unresolved).
    ("MIS Project Template.docx", "mis_project_template.txt", extract_docx),
    ("f1-0-8-shrimp-disease-project-pytorch.ipynb", "reference_notebook.txt", extract_ipynb),
    ("A SYSTEM FOR MODELING SHRIMP INCOME AND DISEASE RISK PREVELENCE USING  MACHINE LEARNING "
     "AND DEEP LEARNING MODELS.pptx", "reference_deck_28slide.txt", extract_pptx),
    ("Shrimp[1].pptx", "reference_deck_18slide.txt", extract_pptx),
    # The two foundational papers, obtained 2026-08-27. Extracted so the passages the project
    # quotes -- the Just-Pope specification, and the within-class density assumption Saerens
    # et al. actually state -- can be checked without reopening a PDF.
    ("Adjusting the outputs of a classifier to new a priori probabilities-a simple procedure.pdf", "saerens_2002.txt", extract_pdf),
    ("Production function estimation and related risk considerations. Most likely American Journal of Agricultural Economics.pdf", "just_pope_1979.txt", extract_pdf),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0

    for filename, out_name, fn in JOBS:
        src = PROJECT_ROOT / filename
        if not src.exists():
            print(f"  MISSING  {filename}")
            failures += 1
            continue
        try:
            body = fn(src)
        except Exception as exc:  # noqa: BLE001 - report and continue, one bad input is not fatal
            print(f"  FAILED   {out_name}: {exc}")
            failures += 1
            continue
        (OUT / out_name).write_text(HEADER.format(src=filename) + body, encoding="utf-8")
        print(f"  wrote    data/interim/extracted_text/{out_name}  ({len(body):,} chars)")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
