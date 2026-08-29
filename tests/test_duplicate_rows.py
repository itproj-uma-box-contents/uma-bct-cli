import unittest

import pandas as pd

from constants import OUTPUT_COLUMNS
from main import check_duplicate_rows


class CheckDuplicateRowsTest(unittest.TestCase):
    def test_flags_every_row_with_a_duplicate_node_title(self):
        rows = [
            {
                "Node ID": "",
                "Node Type": "Unit",
                "Node Title": "1974.0056 Unit 0001",
                "Description": "AU; ARGE Reports; 1966-71",
                "Date Range": "1966-1971;1966-01-01;1971-12-31",
                "Image": "20260815_39.jpg",
                "Flags": "",
            },
            {
                "Node ID": "",
                "Node Type": "Unit",
                "Node Title": "1974.0056 Unit 0002",
                "Description": "AUC; 1967-8; 1973-8; etc; etc; Development; 1966-67",
                "Date Range": "1966-1978;1966-01-01;1978-12-31",
                "Image": "20260815_39.jpg",
                "Flags": "",
            },
            {
                "Node ID": "",
                "Node Type": "Unit",
                "Node Title": "1974.0056 Unit 0001",
                "Description": "AU; ARGE Reports; 1966-71",
                "Date Range": "1966-1971;1966-01-01;1971-12-31",
                "Image": "20260815_40.jpg",
                "Flags": "",
            },
        ]
        records = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

        result = check_duplicate_rows(records)

        duplicate_rows = result["Node Title"] == "1974.0056 Unit 0001"
        self.assertTrue(
            result.loc[duplicate_rows, "Flags"].eq("Potential Dupelicate").all()
        )
        self.assertEqual(result.loc[~duplicate_rows, "Flags"].iloc[0], "")
        self.assertEqual(result.index.name, "Node Title")
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
