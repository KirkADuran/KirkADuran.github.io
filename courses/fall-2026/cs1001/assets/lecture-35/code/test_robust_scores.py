"""Demonstrates both unittest.assertRaises syntaxes plus message checking."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from robust_scores import MalformedScoreError, load_scores, parse_score


class ScoreTests(unittest.TestCase):
    def test_normal_return(self):
        self.assertEqual(("Ada", 87), parse_score("Ada,87"))

    def test_bad_integer_function_call_syntax(self):
        # Syntax A: pass the function object and its arguments.
        self.assertRaises(MalformedScoreError, parse_score, "Ada,nope")

    def test_bad_range_context_manager_syntax(self):
        # Syntax B: put the failing call inside the with block.
        with self.assertRaises(MalformedScoreError):
            parse_score("Ada,150")

    def test_message_and_exception_object(self):
        with self.assertRaisesRegex(MalformedScoreError, "out of range") as cm:
            parse_score("Ada,150")
        self.assertIn("range", str(cm.exception))

    def test_wrong_number_of_fields(self):
        with self.assertRaises(MalformedScoreError):
            parse_score("Ada")

    def test_missing_name(self):
        self.assertRaises(MalformedScoreError, parse_score, ",90")

    def test_exception_chaining(self):
        with self.assertRaises(MalformedScoreError) as cm:
            parse_score("Ada,word")
        self.assertIsInstance(cm.exception.__cause__, ValueError)

    def test_load_valid_file(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            path.write_text("Ada,87\nLin,90\n", encoding="utf-8")
            self.assertEqual([("Ada", 87), ("Lin", 90)], load_scores(str(path)))

    def test_missing_file_is_an_os_error(self):
        with TemporaryDirectory() as folder:
            missing = Path(folder) / "missing.csv"
            with self.assertRaises(OSError):
                load_scores(str(missing))


if __name__ == "__main__":
    unittest.main()
