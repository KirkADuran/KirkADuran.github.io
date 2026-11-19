# CS 1001 — Lecture 35: Exceptions and Error Handling

**Fall 2026 | 50 minutes | 46 slides | 9 original instructional diagrams**

## Contents

- [Editable Beamer source](../../lectures/lecture-35-exceptions-validation-and-file-formats.tex) using the shared course preamble.
- [Published slides](../../pdfs/lecture-35-exceptions-validation-and-file-formats.pdf) rebuilt from the integrated source.
- [Supplied PowerPoint](lecture-35-supplied.pptx) with presenter notes.
- `*.png` contains the nine original diagrams used by the Beamer slides.
- [Python examples ZIP](lecture-35-code.zip) contains the scripts, unit tests, and sample CSV files in `code/`.
- `code/robust_scores.py`: working exception-handling example with a custom exception, a propagating loader, and a CLI boundary handler.
- `code/test_robust_scores.py`: nine unit tests, including both forms of `assertRaises`, `assertRaisesRegex`, and checking `__cause__`.
- `code/exception_variations.py`: multiple `except`, `as exc`, `else`, `finally`, and re-raising.
- `code/scores_good.csv` and `code/scores_bad.csv`: demo input files.

## Beamer compilation

From the repository root, with `latexmk` on `PATH`:

```bash
cd courses/fall-2026/cs1001/lectures
lecture=lecture-35-exceptions-validation-and-file-formats
build_dir="../../build/cs1001/$lecture"
mkdir -p "$build_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" "$lecture.tex" &&
  cp "$build_dir/$lecture.pdf" ../pdfs/
```

The source loads diagrams from `../assets/lecture-35/`.

## Suggested 50-minute pacing

- **0–5 min:** Motivation, exceptions versus ordinary results, Python-raised errors.
- **5–20 min:** `try`/`except`, specific and multiple handlers, generic handlers, messages, `else`, `finally`.
- **20–31 min:** Propagation, deliberate `raise`, re-raise, `raise ... from`.
- **31–39 min:** Defining and using custom exception classes.
- **39–48 min:** Both `assertRaises` syntaxes, exception messages, common mistakes, mini-capstone.
- **48–50 min:** Exit ticket and synthesis.

## Example commands

```bash
cd code
python -m unittest -v test_robust_scores.py
python robust_scores.py scores_good.csv
python robust_scores.py scores_bad.csv
python robust_scores.py missing.csv
python exception_variations.py
```

Parsing and file errors propagate to the score importer's top-level handler. The examples use the Python standard library.

Imported from `lecture35_complete.zip`.
