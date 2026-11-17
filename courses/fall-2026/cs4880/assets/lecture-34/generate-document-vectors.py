#!/usr/bin/env python3
"""Generate the bag-of-words and TF-IDF teaching figures for Lecture 34.

Uses raw term counts and ln(N / df), without smoothing or normalization.
The three-document corpus is deliberately separate from the n-gram example.
"""
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
DOCUMENTS = ["the cat cat", "the dog", "the robot"]
VOCABULARY = ["the", "cat", "dog", "robot"]
GREEN, INK, MUTED, SOFT, GOLD = (
    "#154734", "#263A32", "#63756B", "#EAF2ED", "#B88A2B"
)


def main():
    counts = np.array([
        [Counter(document.split())[term] for term in VOCABULARY]
        for document in DOCUMENTS
    ])
    document_frequency = (counts > 0).sum(axis=0)
    weights = counts * np.log(len(DOCUMENTS) / document_frequency)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": MUTED,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    figure, axis = plt.subplots(figsize=(6.8, 3.0))
    axis.imshow(counts, cmap=LinearSegmentedColormap.from_list(
        "counts", ["#F8FAF8", SOFT, GREEN]), vmin=0, vmax=2, aspect="auto")
    axis.set_xticks(range(len(VOCABULARY)), VOCABULARY, fontsize=14)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(DOCUMENTS)), [
        f'D{i + 1}: "{document}"' for i, document in enumerate(DOCUMENTS)
    ], fontsize=12)
    axis.tick_params(length=0, pad=9)
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            value = counts[row, column]
            axis.text(column, row, str(value), ha="center", va="center",
                      fontsize=20, weight="bold",
                      color="white" if value == 2 else GREEN)
    axis.set_xticks(np.arange(-0.5, counts.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, counts.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=3)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    figure.subplots_adjust(left=0.39, right=0.98, top=0.79, bottom=0.13)
    figure.text(0.5, 0.015, "Each row is one document; each column is one term.",
                ha="center", fontsize=11, color=MUTED)
    figure.savefig(ROOT / "bag-of-words.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.8, 3.0))
    bars = axis.bar(VOCABULARY, weights[0], color=[MUTED, GOLD, GREEN, GREEN],
                    width=0.57, zorder=3)
    axis.set_ylim(0, 2.8)
    axis.set_ylabel("TF-IDF weight", fontsize=12)
    axis.set_title('D1: "the cat cat"', fontsize=15, weight="bold", pad=13)
    axis.set_yticks([0, 1, 2])
    axis.grid(axis="y", alpha=0.16, zorder=0)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    for bar, value in zip(bars, weights[0]):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.10,
                  f"{value:.2f}" if value else "0", ha="center",
                  fontsize=15, weight="bold", color=GREEN)
    axis.tick_params(axis="x", labelsize=14, length=0, pad=8)
    figure.subplots_adjust(left=0.13, right=0.97, top=0.77, bottom=0.23)
    figure.text(0.54, 0.02, 'df(the) = 3 documents; df(cat) = 1 document',
                ha="center", fontsize=11, color=MUTED)
    figure.savefig(ROOT / "tf-idf.png", dpi=200)
    plt.close(figure)
    print("Vocabulary:", VOCABULARY)
    print("Counts:", counts.tolist())
    print("Document frequencies:", document_frequency.tolist())
    print("D1 TF-IDF:", weights[0].round(6).tolist())


if __name__ == "__main__":
    main()
