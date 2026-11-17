"""Lecture 34 demonstration: walk a directory tree and locate .txt files."""
import os
from pathlib import Path


def main() -> None:
    for root, dirs, files in os.walk("data"):
        for name in files:
            if name.endswith(".txt"):
                full_path = Path(root) / name
                print(full_path)


if __name__ == "__main__":
    main()
