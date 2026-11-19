"""CS 1001 Lecture 35: exception handling in a tiny score importer.

Run with:
    python robust_scores.py scores_good.csv
    python robust_scores.py scores_bad.csv
    python robust_scores.py missing.csv
"""
from __future__ import annotations
import sys


class MalformedScoreError(ValueError):
    """A score record is missing fields or has an invalid value."""


def parse_score(line: str) -> tuple[str, int]:
    """Parse a CSV-style 'name,score' line; raise on invalid data."""
    parts = line.strip().split(",")
    if len(parts) != 2:
        raise MalformedScoreError("expected name,score")

    name, raw_score = (part.strip() for part in parts)
    if not name:
        raise MalformedScoreError("name cannot be empty")

    try:
        score = int(raw_score)
    except ValueError as exc:
        raise MalformedScoreError(f"score for {name} must be an integer") from exc

    if not 0 <= score <= 100:
        raise MalformedScoreError("score out of range")
    return name, score


def load_scores(filename: str) -> list[tuple[str, int]]:
    """Read scores; let file and record exceptions propagate to the caller."""
    with open(filename, "r", encoding="utf-8") as file:
        return [parse_score(line) for line in file if line.strip()]


def main(args: list[str] | None = None) -> int:
    if args is None:
        args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python robust_scores.py SCORES.csv")
        return 2
    try:
        scores = load_scores(args[0])
    except OSError as exc:
        print("Cannot read file:", exc)
        return 1
    except MalformedScoreError as exc:
        print("Bad score record:", exc)
        return 1

    for name, score in scores:
        print(f"{name}: {score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
