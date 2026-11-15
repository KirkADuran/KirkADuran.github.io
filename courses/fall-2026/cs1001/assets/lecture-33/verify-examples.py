#!/usr/bin/env python3
"""Execute lecture examples and check printed results, input loops, and script entry."""
from pathlib import Path
import contextlib
import io
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "lectures/lecture-33-input-output-and-string-processing.tex"
text = SOURCE.read_text()
frames = dict(re.findall(
    r"\\begin\{frame\}(?:\[[^\]]*\])?\{([^\n]+)\}\n(.*?)\\end\{frame\}",
    text, re.S))


def blocks(title):
    return re.findall(
        r"\\begin\{Verbatim\}(?:\[[^\]]*\])?\n(.*?)\\end\{Verbatim\}",
        frames[title], re.S)


checked = set()


def run(title, args=(), inputs=(), name="__main__"):
    code = blocks(title)[0]
    namespace = {"__name__": name}
    output = io.StringIO()
    with patch.object(sys, "argv", ["example.py", *args]), \
            patch("builtins.input", side_effect=inputs), \
            contextlib.redirect_stdout(output):
        exec(compile(code, f"{SOURCE.name}:{title}", "exec"), namespace)
    checked.add(title)
    return output.getvalue(), namespace


# Compare the source code with the source's own expected-output blocks.
for title in ["print() Produces Console Output", "Formatting Readable Output",
              "Common String Operations for Input", "Decompose a Pair Parser"]:
    assert run(title)[0] == blocks(title)[1], title

assert run("Three Concepts Worth Separating")[0] == "8\n"
out, scope = run("Predict: Return Versus Print")
assert out == blocks("Reveal: Console Output Is Not a Return Value")[0]
assert scope["result"] is None
assert run("Predict: Token Types")[0] == blocks("Reveal: Concatenation Versus Arithmetic")[0]
assert run("input() Waits for a Response", inputs=["Ava"])[0] == "Hello, Ava!\n"
assert run("Converting Text to Numbers", inputs=["3.5"])[0] == "Twice that value is 7.0\n"
assert run("Build a Useful Message from Input", inputs=[" ada ", " LOVELACE "])[0] == "Ada Lovelace\n"

sum_pair = run("Decompose a Pair Parser")[1]["sum_pair"]
assert sum_pair("\t-3.5  2\n") == -1.5
assert sum_pair("0 0") == 0.0
keyboard_sum = run("An Input Loop with a Sentinel")[1]["keyboard_sum"]
for values, expected in [(["2.5", "-1", "4", "done"], 5.5), (["done"], 0.0),
                         (["0", "-2", "2", "done"], 0.0)]:
    with patch("builtins.input", side_effect=values) as user_input:
        assert keyboard_sum() == expected
        assert user_input.call_count == len(values)

out, scope = run("Inspect sys.argv", args=["3.5", "2", "-1"])
assert out == "['example.py', '3.5', '2', '-1']\n"
assert run("Numbers in Arguments Are Still Strings", args=["6", "7"])[0] == "67\n13\n"
assert run("Avoid Missing-Argument Index Errors")[0] == "Please provide a name\n"
assert run("Avoid Missing-Argument Index Errors", args=["Ada Lovelace"])[0] == "Hello, Ada Lovelace!\n"
assert run("Loop Over User Arguments", args=["3.5", "2", "-1"])[0] == "4.5\n"
assert run("Loop Over User Arguments")[0] == "0.0\n"
assert run("A main() Function Expresses the Entry Point", inputs=["Ava"])[0] == "Hello, Ava!\n"

for title, expected in [("The Standard Script Pattern", "Program started\n"),
                        ("Predict: Running Versus Importing", "Hi, Ava\n")]:
    assert run(title)[0] == expected
    out, scope = run(title, name="imported_example")
    assert out == "" and callable(scope["main"])
assert scope["greet"]("Bo") == "Hi, Bo"

title = "Integrated Example: CLI Number Sum"
for args, expected in [(["3.5", "2", "-1"], "Total: 4.5\n"), ([], "Total: 0.0\n"),
                       (["-2", "2", "0"], "Total: 0.0\n")]:
    assert run(title, args=args)[0] == expected
out, scope = run(title, name="imported_example")
assert out == "" and scope["sum_numbers"](["1.5", "-2"]) == -0.5

# The deliberate bug must fail even when a numeric argument is provided.
try:
    run("Think-Pair-Share: Find the Bug", args=["6"])
except ValueError:
    checked.add("Think-Pair-Share: Find the Bug")
else:
    raise AssertionError("The deliberate argv[0] bug no longer raises ValueError")
assert run("Reveal: Select and Check the Correct Argument")[0] == "Missing numeric argument\n"
assert run("Reveal: Select and Check the Correct Argument", args=["6"])[0] == "Amount: 6.0\n"

# Check the complete script through real interpreter and import boundaries.
with tempfile.TemporaryDirectory(prefix="cs1001-l33-") as directory:
    root = Path(directory)
    (root / "total.py").write_text(blocks("Integrated Example: CLI Number Sum")[0])
    (root / "greetings.py").write_text(blocks("Predict: Running Versus Importing")[0])
    for args, expected in [(["total.py", "3.5", "2", "-1"], "Total: 4.5\n"),
                           (["total.py"], "Total: 0.0\n"),
                           (["greetings.py"], "Hi, Ava\n"),
                           (["-c", "import greetings, total"], ""),
                           (["-c", "import greetings; print(greetings.greet('Bo'))"], "Hi, Bo\n")]:
        result = subprocess.run([sys.executable, *args], cwd=root, text=True,
                                capture_output=True, check=True)
        assert result.stdout == expected and result.stderr == "", (args, result)

print(f"PASS: {len(checked)} executable slide examples; expected outputs, sentinel boundaries,")
print("CLI parsing, deliberately broken indexing, and direct execution versus import.")
