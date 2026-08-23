# Fall 2026 Course Slides

This directory contains placeholder Beamer slide decks for Kirk Duran's Fall 2026 courses.

## Courses

- [CS 1001: Introduction to Computing](cs1001/)
- [CS 4880: Artificial Intelligence](cs4880/)

## Build PDFs

From this directory:

```sh
make
```

The Makefile compiles every numbered lecture source in `cs1001/lectures/` and `cs4880/lectures/` into course-specific `pdfs/` directories. It expects `latexmk` to be available on `PATH`, or in `~/Library/TinyTeX/bin/universal-darwin`.

## Workflow

1. Edit the lecture `.tex` source.
2. Run `make` from `courses/fall-2026/`.
3. Commit both the source `.tex` and generated `.pdf` for GitHub Pages.
