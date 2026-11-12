# Anonymous CVPR 2027 working draft

## Build the paper

From the extracted archive root, run the following with a TeX Live installation that provides the CVPR template's LaTeX packages and `latexmk`.

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex supplement.tex
```

The anonymous PDFs are `build/main.pdf` and `build/supplement.pdf`. The main paper has eight pages of text and figures before the LLM Usage Statement and references. The LLM Usage Statement is included in both documents.

## Regenerate tables and figures

One command rebuilds all tables, all figures and both PDFs: `./scripts/make_all.sh`. The revised Times-font figures are drawn by `scripts/` (see `REVISION_NOTES.md`); the original renderer below still produces the tables.

### Original renderer

The archive contains the vector figure PDFs and generated LaTeX tables. To regenerate them, use Python 3.9–3.11 and run the following from the extracted archive root.

```sh
python -m pip install -r repro/requirements.txt
python repro/paper_assets.py --paper .
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex supplement.tex
```

`repro/aggregate.json` contains only aggregate metrics, patient intervals, counts and fitted-model summaries. It includes all seven matched rates, the original adjusted estimates, all fixed-threshold comparisons, cap and subset checks, the contour-group sensitivity estimates, and new development and evaluation summaries. `original` and `new` distinguish the two locks. Values are fractions unless the field specifies a count, a physical unit or FP/scan. This archive rebuilds the paper and its assets independently of private data or training outputs.
