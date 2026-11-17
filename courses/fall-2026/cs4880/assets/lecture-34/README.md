# CS 4880 - Lecture 34: Natural Language Processing

**Class date:** November 18, 2026\
**Duration:** Approximately 50 minutes\
**Source format:** LaTeX Beamer, 16:9 (matches Lectures 31-33)\
**Slide count:** 45

## Storyline

1. Why language is challenging, and historical motivation (6 min)
2. Dartmouth, symbolic grammar, parse trees, and ELIZA (9 min)
3. Tokenization, Shannon, and n-gram prediction (12 min)
4. One-hot vectors, distributional meaning, and word2vec (11 min)
5. Sequence models, encoder-decoder seq2seq, and failure modes (9 min)
6. Attention motivation and exit ticket for the next lecture (3 min)

Every substantive method has a corresponding failure or limitation. Historical references and direct paper links are in the last two slides. All 27 PNG illustrations are original, generated programmatically for this lecture. Examples involving embedding geometry or particular subword splits are explicitly illustrative, not trained-model outputs.

## Compile from this project

```bash
cd courses/fall-2026/cs4880/lectures
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../build/cs4880/lecture-34-natural-language-processing-language-representation-sequence-mod \
  lecture-34-natural-language-processing-language-representation-sequence-mod.tex
cp ../../build/cs4880/lecture-34-natural-language-processing-language-representation-sequence-mod/*.pdf ../pdfs/
```

The main document imports the supplied lecture-specific `../assets/lecture-34/preamble.tex` and loads images from `../assets/lecture-34/`. The shared course preamble is unchanged. The PDF in `cs4880/pdfs/` is published by the course index.

## Regenerate figures

```bash
cd courses/fall-2026/cs4880/assets/lecture-34
python -m pip install -r requirements.txt
python generate-figures.py
```

The figure script uses only NumPy and Matplotlib. The small five-sentence corpus is included in the script so the teaching calculation can be reproduced.

## Notes

Imported from `CS4880_Lecture34_NLP_Source_Project.zip`. The `.tex` file is the fully editable source; the archive contains no PDF or PowerPoint export. Supplied figures are retained unchanged.
