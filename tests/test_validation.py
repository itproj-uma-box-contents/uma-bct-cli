import unittest

import pandas as pd

from uma_bct.constants import LOW_CONFIDENCE_FLAG, OUTPUT_COLUMNS
from uma_bct.validation import (
    check_low_confidence_rows,
    escape_spreadsheet_formula,
)


class CheckLowConfidenceRowsTest(unittest.TestCase):
    def test_requires_confidence_and_flag_columns(self):
        with self.assertRaisesRegex(ValueError, "Confidence Score"):
            check_low_confidence_rows(pd.DataFrame({"Flags": []}))

    def test_flags_only_scores_strictly_below_eighty_by_default(self):
        dataframe = pd.DataFrame(
            [
                self.row(79),
                self.row(80, "Potential Duplicate"),
                self.row(91),
                self.row("not supplied"),
            ],
            columns=OUTPUT_COLUMNS,
        )

        result = check_low_confidence_rows(dataframe)

        self.assertEqual(
            result["Flags"].tolist(),
            [LOW_CONFIDENCE_FLAG, "Potential Duplicate", "", ""],
        )

    def test_preserves_existing_flags_without_repeating_low_confidence(self):
        dataframe = pd.DataFrame(
            [self.row(12, "Missing unit; Low confidence")],
            columns=OUTPUT_COLUMNS,
        )

        result = check_low_confidence_rows(dataframe)

        self.assertEqual(result.loc[0, "Flags"], "Missing unit; Low confidence")

    def test_rejects_an_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            check_low_confidence_rows(pd.DataFrame(columns=OUTPUT_COLUMNS), 101)

    def test_escapes_formula_prefix_after_leading_whitespace(self):
        self.assertEqual(escape_spreadsheet_formula("=1+1"), "'=1+1")
        self.assertEqual(escape_spreadsheet_formula("\t-2+3"), "'\t-2+3")
        self.assertEqual(escape_spreadsheet_formula("ordinary text"), "ordinary text")
        self.assertEqual(escape_spreadsheet_formula("'@safe"), "'@safe")

    @staticmethod
    def row(confidence: object, flags: str = "") -> dict[str, object]:
        return {
            "Node ID": "",
            "Node Type": "Unit",
            "Node Title": "1974.0056 Unit 0001",
            "Description": "Archive box contents",
            "Date Range": "",
            "Confidence Score": confidence,
            "Flags": flags,
            "Image": "box.jpg",
        }


if __name__ == "__main__":
    unittest.main()
