# CS4880 Lecture 35: Transformers — From Tokens to Attention

- **Lecture date** November 20, 2026
- **Duration** Approximately 50 minutes
- [Editable Beamer source](../../lectures/lecture-35-transformers-and-large-language-models-attention-transformers-pr.tex)

This 43-slide Beamer lecture follows the style of Lectures 31–34. The lesson combines and restructures the two user-supplied lecture PDFs while distinguishing their source descriptions from updates and technical clarifications. It covers the pipeline from text to logits, Q/K/V, an exact numerical attention example, multi-head attention, masking, model variants, residual blocks, training, generation, and limitations.

## Build

From `courses/fall-2026/cs4880/lectures/`, with `latexmk` available on `PATH`:

```bash
lecture=lecture-35-transformers-and-large-language-models-attention-transformers-pr
build_dir="../../build/cs4880/$lecture"
mkdir -p "$build_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$build_dir" "$lecture.tex" &&
  cp "$build_dir/$lecture.pdf" ../pdfs/
```

The source loads the figures and its Cal Poly green Beamer preamble from `../assets/lecture-35/`. The course index links to the built PDF.

## Regenerate 23 images

From `courses/fall-2026/cs4880/assets/lecture-35/`:

```bash
python3 -m pip install -r requirements.txt
python3 generate-figures.py
```

The generator uses NumPy and Matplotlib. It computes the worked attention example from its inputs and labels schematic diagrams.

## Export PowerPoint

From `courses/fall-2026/cs4880/assets/lecture-35/`, with Node.js and Poppler installed:

```bash
export_dir=$(mktemp -d)
npm install --prefix "$export_dir" pptxgenjs
pdftoppm -r 144 -png \
  ../../pdfs/lecture-35-transformers-and-large-language-models-attention-transformers-pr.pdf \
  "$export_dir/page"
NODE_PATH="$export_dir/node_modules" node assemble-pptx.js \
  "$export_dir" "$export_dir/CS4880_Lecture35_Transformers.pptx"
```

The PowerPoint places one rendered PDF page on each slide. The TeX file is the editable deck.

## Sources

- User-supplied: `25_480_W26.pdf` (29-page attention / transformer lecture)
- User-supplied: `Copy of 24_480_W26.pdf` (10-page LLM pipeline lecture)
- Bahdanau, Cho, and Bengio (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*.
- Vaswani et al. (2017), *Attention Is All You Need*.
- Devlin et al. (2019), *BERT*; Radford et al. (2018), *GPT*.

Imported from `CS4880_Lecture35_Source_Project.zip`.
