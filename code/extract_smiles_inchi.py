#!/usr/bin/env python3
# extract_smiles_inchi.py
#
# usage:
#   python extract_smiles_inchi.py input.sdf output.csv
#
# Each row in the CSV corresponds to one compound in the SDF.
# Columns: smiles , inchi

import re
import csv
import sys
from pathlib import Path

SMILES_RE = re.compile(r"\bsmiles\s*=\s*([^\s;]+)", re.IGNORECASE)
INCHI_RE  = re.compile(r"\bInChI\s*=\s*(InChI=[^\s;]+)", re.IGNORECASE)
COMMENT_BLOCK_RE = re.compile(r"<COMMENT>\s*(.*?)\s*(?:<|$)", re.DOTALL)

def extract_from_comment(comment: str):
    """Return (smiles, inchi) strings or '' if not found."""
    smiles_match = SMILES_RE.search(comment)
    inchi_match  = INCHI_RE.search(comment)
    smiles = smiles_match.group(1) if smiles_match else ""
    inchi  = inchi_match.group(1)  if inchi_match  else ""
    return smiles, inchi

def parse_sdf(path: Path):
    """Yield (smiles, inchi) for every record in the SDF file."""
    text = path.read_text(errors="ignore")
    for record in text.split("$$$$"):
        m = COMMENT_BLOCK_RE.search(record)
        if not m:
            # no <COMMENT> section → empty columns
            yield "", ""
            continue
        yield extract_from_comment(m.group(1))

def main(inp: str, outp: str):
    rows = list(parse_sdf(Path(inp)))
    # write CSV
    with open(outp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "inchi"])        # header
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {outp}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python extract_smiles_inchi.py input.sdf output.csv")
    main(sys.argv[1], sys.argv[2])
