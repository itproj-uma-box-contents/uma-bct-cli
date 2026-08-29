import unittest

import pandas as pd

from constants import OUTPUT_COLUMNS
from src.validation import LOW_CONFIDENCE_FLAG, check_low_confidence_rows


class CheckLowConfidenceRowsTest(unittest.TestCase):
    def test_flags_only_scores_strictly_below_fifty(self):
        dataframe = pd.DataFrame(
            [
                self.row(49),
                self.row(50, "Potential Duplicate"),
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

    @staticmethod
    def row(confidence: object, flags: str = "") -> dict[str, object]:
        return {
            "Node": "Unit",
            "Title": "1974.0056 Unit 0001",
            "Description": "Archive box contents",
            "Start Date": "",
            "End Date": "",
            "Confidence Score": confidence,
            "Flags": flags,
            "Image": "box.jpg",
        }


if __name__ == "__main__":
    unittest.main()
