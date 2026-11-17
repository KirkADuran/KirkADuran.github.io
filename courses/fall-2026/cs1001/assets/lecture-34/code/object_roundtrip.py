"""Lecture 34 demonstration: save attributes and reconstruct new objects."""
import json
from pathlib import Path


class Student:
    def __init__(self, name: str, score: int):
        self.name = name
        self.score = score


def main() -> None:
    students = [Student("Ada", 93), Student("Lin", 87)]
    records = [{"name": s.name, "score": s.score} for s in students]

    with Path("saved_students.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    with Path("saved_students.json").open("r", encoding="utf-8") as f:
        rows = json.load(f)

    recovered = [Student(row["name"], row["score"]) for row in rows]
    print([(s.name, s.score) for s in recovered])
    print("Different objects?", recovered[0] is not students[0])


if __name__ == "__main__":
    main()
