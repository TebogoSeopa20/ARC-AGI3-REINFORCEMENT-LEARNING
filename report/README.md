# Report

Download the official RLC 2026 LaTeX template (linked from the RLC submission instructions) and copy its
style files into this folder. Keep its fonts, margins and spacing; disable the cover page as its
instructions describe. `main.tex` holds the section structure and figure/table slots only — replace the
`\documentclass`/`\usepackage` lines with the template's own preamble.

Limit: 8 pages main content (figures, tables and pre-reference appendices included). References excluded.
Build: `latexmk -pdf main.tex`. Figures and tables come from `outputs/figures/` (`scripts/aggregate.py`).
