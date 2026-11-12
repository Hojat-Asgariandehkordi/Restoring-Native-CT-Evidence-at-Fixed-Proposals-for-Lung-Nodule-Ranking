#!/bin/sh
# Rebuild every table and figure, then both PDFs. Run from the repository root.
# Order matters: repro/paper_assets.py writes the original tables (and older-style figures);
# scripts/ then overwrites the figures with the revised Times-font versions.
set -e
python3 repro/paper_assets.py --paper .
python3 scripts/tables.py
python3 scripts/fig_schematic.py
python3 scripts/fig_results.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex supplement.tex
