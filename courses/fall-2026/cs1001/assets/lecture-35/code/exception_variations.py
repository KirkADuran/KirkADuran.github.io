"""Small standalone demonstrations from Lecture 35."""

from __future__ import annotations


def divide_text(raw: str) -> float | None:
    try:
        denominator = int(raw)
        value = 100 / denominator
    except ValueError as exc:
        print("Expected an integer:", exc)       # specific + message
        return None
    except ZeroDivisionError:
        print("Cannot divide by zero")             # second specific type
        return None
    else:
        print("Calculation worked")                 # no exception
        return value
    finally:
        print("Calculation attempt finished")      # always for ordinary exits


def logged_conversion(raw: str) -> int:
    try:
        return int(raw)
    except ValueError:
        print("Logging a failed conversion")
        raise                                     # re-raise active error


if __name__ == "__main__":
    for raw in ("4", "0", "nope"):
        print("Input:", repr(raw), "Result:", divide_text(raw))
