import unittest

import pandas as pd

from uma_bct.constants import OUTPUT_COLUMNS
from uma_bct.duplicates import check_duplicate_rows


class CheckDuplicateRowsTest(unittest.TestCase):
    def test_requires_title_and_flag_columns(self):
        with self.assertRaisesRegex(ValueError, "Node Title"):
            check_duplicate_rows(pd.DataFrame({"Flags": []}))

    def test_flags_every_row_with_a_duplicate_canonical_title(self):
        records = pd.DataFrame(
            [
                self.row("1974.0056 Unit 0001", "first.jpg"),
                self.row("1974.0056 Unit 0002", "second.jpg"),
                self.row("1974.0056 Unit 0001", "third.jpg"),
            ],
            columns=OUTPUT_COLUMNS,
        )

        result = check_duplicate_rows(records)

        duplicate_rows = result["Node Title"] == "1974.0056 Unit 0001"
        self.assertTrue(
            result.loc[duplicate_rows, "Flags"].eq("Potential Duplicate").all()
        )
        self.assertEqual(result.loc[~duplicate_rows, "Flags"].iloc[0], "")
        self.assertEqual(result.index.name, "Node Title")
        self.assertEqual(len(result), 3)

    def test_preserves_existing_flags_and_does_not_repeat_duplicate_flag(self):
        records = pd.DataFrame(
            [
                self.row("1974.0056 Unit 0001", "first.jpg", "Low confidence"),
                self.row(
                    "1974.0056 Unit 0001",
                    "second.jpg",
                    "Potential Duplicate",
                ),
            ],
            columns=OUTPUT_COLUMNS,
        )

        result = check_duplicate_rows(records)

        self.assertEqual(
            result["Flags"].tolist(),
            ["Low confidence; Potential Duplicate", "Potential Duplicate"],
        )

    def test_compares_canonical_titles_case_insensitively(self):
        records = pd.DataFrame(
            [
                self.row("UMA-AC-000000330 Unit 0001", "first.jpg"),
                self.row("uma-ac-000000330 unit 0001", "second.jpg"),
            ],
            columns=OUTPUT_COLUMNS,
        )

        result = check_duplicate_rows(records)

        self.assertTrue(result["Flags"].eq("Potential Duplicate").all())

    def test_does_not_treat_missing_or_blank_titles_as_duplicates(self):
        records = pd.DataFrame(
            [
                self.row("", "first.jpg"),
                self.row("   ", "second.jpg"),
                self.row(None, "third.jpg"),
                self.row(None, "fourth.jpg"),
            ],
            columns=OUTPUT_COLUMNS,
        )

        result = check_duplicate_rows(records)

        self.assertTrue(result["Flags"].eq("").all())

    @staticmethod
    def row(node_title: str | None, image: str, flags: str = "") -> dict[str, object]:
        return {
            "Node ID": "",
            "Node Type": "Unit",
            "Node Title": node_title,
            "Description": "Archive box contents",
            "Date Range": "",
            "Confidence Score": 90,
            "Flags": flags,
            "Image": image,
        }


if __name__ == "__main__":
    unittest.main()
