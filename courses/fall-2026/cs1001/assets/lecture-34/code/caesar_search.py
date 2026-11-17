"""Lecture 34 capstone: recover a Caesar-shifted message using a known word.

Run this in a directory containing secret.txt:
    python caesar_search.py

This intentionally uses only uppercase A-Z and preserves spaces/punctuation.
Caesar ciphers are not secure encryption.
"""
from pathlib import Path


def decode_caesar(text: str, key: int) -> str:
    result = ""
    for ch in text:
        if "A" <= ch <= "Z":
            position = ord(ch) - ord("A")
            result += chr((position - key) % 26 + ord("A"))
        else:
            result += ch
    return result


def find_possible_keys(ciphertext: str, known_word: str) -> list[tuple[int, str]]:
    matches = []
    for key in range(26):
        candidate = decode_caesar(ciphertext, key)
        if known_word.upper() in candidate:
            matches.append((key, candidate))
    return matches


def main() -> None:
    with Path("secret.txt").open("r", encoding="utf-8") as file:
        ciphertext = file.read()

    for key, candidate in find_possible_keys(ciphertext, "BRIDGE"):
        print(f"Possible key {key}: {candidate.strip()}")


if __name__ == "__main__":
    main()
