# CS 1001 Lecture 33: Input, Output, and String Processing

Imported from `Lecture_33_Complete.zip`: 45 slides and seven instructional diagrams. The lecture covers console output, keyboard input, string parsing, sentinel loops, command-line arguments, and script organization with `main()` and the name guard. File I/O and malformed-input exception handling are deferred to later lectures.

The TeX uses the shared course preamble, the existing November 16, 2026 lecture date and course unit, and the diagrams in this directory. The sentinel-loop diagram is centered with an explanatory callout instead of an empty second column. The published PDF is rebuilt from the integrated source.

`lecture-33-supplied.pptx` is the original editable PowerPoint, retained unchanged with its speaker notes. It reflects the supplied version rather than the repository's theme and layout adjustments. It is linked from the lecture's course-index row alongside the TeX source.

## Build

From `courses/fall-2026/cs1001/lectures/`, with TinyTeX on `PATH`:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../build/cs1001/lecture-33-input-output-and-string-processing \
  lecture-33-input-output-and-string-processing.tex
```

Copy the resulting PDF to `cs1001/pdfs/lecture-33-input-output-and-string-processing.pdf` when publishing an update.

## Verify examples

From this directory:

```sh
python3 verify-examples.py
```

The verifier extracts examples directly from the TeX and checks the stated output, text-to-number conversion, immediate and repeated sentinel input, missing and numeric command-line arguments, the intentional indexing bug, and entry-point behavior. Complete scripts are also run and imported with a real Python interpreter. It uses only the Python standard library.

## Suggested 50-minute flow

1. Console boundaries, `print()`, and output versus return.
2. Keyboard input and numeric conversion.
3. String methods and sentinel loops.
4. Command-line arguments and `sys.argv`.
5. `main()`, the name guard, helper functions, and integrated examples.

Prediction and reveal slides provide discussion pauses. The supplied PowerPoint includes speaker notes with pacing suggestions.
