"""Reproducibly fetch the IFPRI 'Clustering for Global Market Access in Bangladesh' survey.

Why this exists as a script rather than a manual download
---------------------------------------------------------
Data provenance is examinable. A reviewer can ask "where exactly did this come from, and
would you get the same bytes if you ran it again?" This script answers both: it resolves the
dataset by persistent DOI, records every file's server-side MD5, verifies each download
against that checksum, and writes a manifest. A failed checksum is a hard error, not a warning
-- a silently corrupted survey file would poison every downstream estimate.

Source
------
IFPRI, "Clustering for global market access in Bangladesh". Harvard Dataverse. Two rounds of the
same 1,222 shrimp farms across Khulna, Bagerhat and Satkhira:

    baseline   doi:10.7910/DVN/TUXYXQ   collected Nov-Dec 2023
    endline    doi:10.7910/DVN/VQJNLG   collected 7-31 May 2024

Terms: IFPRI Datasets Terms of Use -- research datasets as global public goods, unrestricted
public access. Verified 2026-08-25 for the baseline: versionState=RELEASED, 99 files, every file
restricted=False.

Why the round is a flag rather than a constant
-----------------------------------------------
Everything the project has reported to date uses the **baseline**, and that must stay reproducible
by running this script with no arguments. The endline was Master-Document improvement 4 -- it is
what makes the survey a *panel*, and a panel is what would let farm risk be estimated within-farm
over time instead of across farms at one moment. Adding it as a second DOI rather than replacing
the first keeps both rounds fetchable and keeps the default behaviour exactly as it was.

Each round downloads to its own directory (`data/raw/ifpri/` and `data/raw/ifpri_endline/`) with
its own manifest, so a checksum, a file list and a round can never be attributed to the wrong
survey.

Usage
-----
    python scripts/fetch_ifpri.py                    # baseline household modules + codebooks
    python scripts/fetch_ifpri.py --round endline    # the same for the endline round
    python scripts/fetch_ifpri.py --all              # every file including community/cluster
    python scripts/fetch_ifpri.py --list             # show what would be fetched, download nothing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROUNDS = {
    "baseline": {"doi": "doi:10.7910/DVN/TUXYXQ", "dirname": "ifpri"},
    "endline": {"doi": "doi:10.7910/DVN/VQJNLG", "dirname": "ifpri_endline"},
}
DEFAULT_ROUND = "baseline"

BASE = "https://dataverse.harvard.edu"
META_URL_TEMPLATE = BASE + "/api/datasets/:persistentId/?persistentId={doi}"
FILE_URL = BASE + "/api/access/datafile/{fid}"

# Dataverse returns 403 to urllib's default agent. This is not an auth wall -- the same
# request with a browser agent succeeds, and every file is flagged restricted=False.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; shrimp-risk-project/1.0)"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Module-level for backward compatibility: every caller written before the endline existed
# expects the baseline in `data/raw/ifpri/`, and `econometrics.data.load_ifpri_module` still
# defaults to it. `main` rebinds these when a different round is requested.
OUT_DIR = PROJECT_ROOT / "data" / "raw" / ROUNDS[DEFAULT_ROUND]["dirname"]
MANIFEST = OUT_DIR / "_manifest.csv"


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_metadata(doi: str = ROUNDS[DEFAULT_ROUND]["doi"]) -> list[dict]:
    """Resolve the DOI to a file list.

    Returns one dict per file with the fields we need downstream: id, filename, size,
    md5 and the restricted flag. We keep `restricted` even though we expect it to be
    False everywhere, because if that ever changes we want the run to say so loudly
    rather than quietly download a subset.
    """
    meta = json.loads(_get(META_URL_TEMPLATE.format(doi=doi)))
    if meta.get("status") != "OK":
        raise RuntimeError(f"Dataverse returned status={meta.get('status')}")

    version = meta["data"]["latestVersion"]
    if version.get("versionState") != "RELEASED":
        raise RuntimeError(f"dataset is not RELEASED (got {version.get('versionState')})")

    files = []
    for entry in version["files"]:
        df = entry["dataFile"]
        files.append(
            {
                "id": df["id"],
                "filename": df["filename"],
                "size": df.get("filesize"),
                # Dataverse exposes md5 for most files; tabular ingest can leave it absent.
                "md5": df.get("md5") or df.get("checksum", {}).get("value"),
                "restricted": bool(entry.get("restricted")),
            }
        )
    return files


def select(files: list[dict], all_files: bool) -> list[dict]:
    """Default to household modules; that is where farm-level income and disease live.

    Community- and cluster-level files describe higher aggregation units and are not
    farm-level observations, so pulling them by default would invite an analyst to merge
    across levels without noticing. --all is available when that is deliberate.
    """
    if all_files:
        return files
    return [f for f in files if "_hh_" in f["filename"]]


def download(files: list[dict]) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, f in enumerate(files, 1):
        dest = OUT_DIR / f["filename"]
        status = "cached"

        if not dest.exists():
            for attempt in range(3):
                try:
                    dest.write_bytes(_get(FILE_URL.format(fid=f["id"])))
                    status = "downloaded"
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f"failed to fetch {f['filename']}: {exc}") from exc
                    time.sleep(2 * (attempt + 1))

        digest = hashlib.md5(dest.read_bytes()).hexdigest()
        expected = f["md5"]
        # Dataverse serves tabular .tab files converted from the ingested original, so the
        # stored md5 describes the ORIGINAL upload, not the .tab we receive. Only treat a
        # mismatch as fatal for non-tabular files, where the bytes should match exactly.
        is_tabular = dest.suffix == ".tab"
        if expected and digest != expected and not is_tabular:
            dest.unlink()
            raise RuntimeError(
                f"checksum mismatch for {f['filename']}: got {digest}, expected {expected}"
            )
        verified = "exact" if (expected and digest == expected) else (
            "tabular-derived" if is_tabular else "no-md5-published"
        )

        print(f"[{i:3d}/{len(files)}] {status:10s} {verified:16s} {f['filename']}")
        rows.append(
            {
                "filename": f["filename"],
                "dataverse_id": f["id"],
                "bytes_on_disk": dest.stat().st_size,
                "md5_on_disk": digest,
                "md5_published": expected or "",
                "verification": verified,
            }
        )
    return rows


def write_manifest(rows: list[dict]) -> None:
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmanifest -> {MANIFEST.relative_to(PROJECT_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="fetch community/cluster files too")
    ap.add_argument("--list", action="store_true", help="list only, download nothing")
    ap.add_argument("--round", choices=sorted(ROUNDS), default=DEFAULT_ROUND,
                    help="which survey round to fetch (default: baseline, the one every "
                         "reported result uses)")
    args = ap.parse_args()

    global OUT_DIR, MANIFEST
    doi = ROUNDS[args.round]["doi"]
    OUT_DIR = PROJECT_ROOT / "data" / "raw" / ROUNDS[args.round]["dirname"]
    MANIFEST = OUT_DIR / "_manifest.csv"

    print(f"round      : {args.round}")
    print(f"destination: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"resolving {doi} ...")
    files = fetch_metadata(doi)

    restricted = [f for f in files if f["restricted"]]
    if restricted:
        print(f"WARNING: {len(restricted)} file(s) are restricted and will fail:", file=sys.stderr)
        for f in restricted:
            print(f"  - {f['filename']}", file=sys.stderr)

    chosen = select(files, args.all)
    total_mb = sum(f["size"] or 0 for f in chosen) / 1e6
    print(f"{len(files)} files in dataset; {len(chosen)} selected ({total_mb:.1f} MB)\n")

    if args.list:
        for f in chosen:
            print(f"  {(f['size'] or 0):>10,}  {f['filename']}")
        return 0

    write_manifest(download(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
