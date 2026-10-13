
import unittest

from uma_bct.util.date_util import Date, DateRange, get_earliest_date, get_latest_date


class TestDateIsValid(unittest.TestCase):
    def test_fully_empty_date_is_valid(self):
        # Defaults to 2000-01-01, which is a real date.
        self.assertTrue(Date().is_valid())

    def test_valid_full_date(self):
        self.assertTrue(Date(day=15, month=6, year=2023).is_valid())

    def test_year_only_is_valid(self):
        self.assertTrue(Date(year=2023).is_valid())

    def test_month_and_day_only_is_valid(self):
        self.assertTrue(Date(day=15, month=6).is_valid())

    def test_day_out_of_range_is_invalid(self):
        # February never has 32 days.
        self.assertFalse(Date(day=32, month=2, year=2023).is_valid())

    def test_month_out_of_range_is_invalid(self):
        self.assertFalse(Date(day=1, month=13, year=2023).is_valid())

    def test_leap_year_feb_29_is_valid(self):
        self.assertTrue(Date(day=29, month=2, year=2024).is_valid())

    def test_non_leap_year_feb_29_is_invalid(self):
        self.assertFalse(Date(day=29, month=2, year=2023).is_valid())


class TestDateRangeBasics(unittest.TestCase):

    def test_default_construction(self):
        dr = DateRange()
        self.assertEqual(dr.start, Date())
        self.assertEqual(dr.end, Date())

    def test_to_string(self):
        dr = DateRange(start=Date(year=2020), end=Date(year=2021))
        self.assertEqual(dr.to_string(), "2020 - 2021")

    def test_invalid_if_start_is_invalid(self):
        dr = DateRange(start=Date(day=32, month=2, year=2023), end=Date(year=2023))
        self.assertFalse(dr.is_valid())

    def test_invalid_if_end_is_invalid(self):
        dr = DateRange(start=Date(year=2023), end=Date(day=32, month=2, year=2023))
        self.assertFalse(dr.is_valid())

    def test_ordered_years_is_valid(self):
        dr = DateRange(start=Date(year=2020), end=Date(year=2021))
        self.assertTrue(dr.is_valid())

    def test_equal_years_is_valid(self):
        dr = DateRange(start=Date(year=2020), end=Date(year=2020))
        self.assertTrue(dr.is_valid())

    def test_reversed_years_is_invalid(self):
        dr = DateRange(start=Date(year=2021), end=Date(year=2020))
        self.assertFalse(dr.is_valid())

    def test_ordered_months_same_year_is_valid(self):
        dr = DateRange(
            start=Date(year=2020, month=3),
            end=Date(year=2020, month=6),
        )
        self.assertTrue(dr.is_valid())

    def test_reversed_months_same_year_is_invalid(self):
        dr = DateRange(
            start=Date(year=2020, month=6),
            end=Date(year=2020, month=3),
        )
        self.assertFalse(dr.is_valid())

    def test_ordered_years_different_digit_lengths_is_valid(self):
        dr = DateRange(start=Date(year=200), end=Date(year=1000))
        self.assertTrue(dr.is_valid())

    def test_ordered_months_different_digit_lengths_is_valid(self):
        dr = DateRange(
            start=Date(year=2020, month=9),
            end=Date(year=2020, month=10),
        )
        self.assertTrue(dr.is_valid())


class TestGetEarliestAndLatestDate(unittest.TestCase):

    def test_empty_tuple_returns_default_date_earliest(self):
        self.assertEqual(get_earliest_date(()), Date())

    def test_empty_tuple_returns_default_date_latest(self):
        self.assertEqual(get_latest_date(()), Date())

    def test_single_date(self):
        d = Date(year=2020)
        self.assertEqual(get_earliest_date((d,)), d)
        self.assertEqual(get_latest_date((d,)), d)

    def test_multiple_dates_same_digit_length(self):
        d1 = Date(year=2019)
        d2 = Date(year=2020)
        d3 = Date(year=2021)
        self.assertEqual(get_earliest_date((d2, d3, d1)), d1)
        self.assertEqual(get_latest_date((d2, d3, d1)), d3)

    def test_date_range_input_is_expanded_to_start_and_end(self):
        # A DateRange should contribute both its start and end to the
        # comparison pool.
        dr = DateRange(start=Date(year=2018), end=Date(year=2022))
        other = Date(year=2020)
        self.assertEqual(get_earliest_date((dr, other)), dr.start)
        self.assertEqual(get_latest_date((dr, other)), dr.end)

    def test_mixed_dates_and_ranges(self):
        d = Date(year=2019)
        dr = DateRange(start=Date(year=2020), end=Date(year=2025))
        self.assertEqual(get_earliest_date((d, dr)), d)
        self.assertEqual(get_latest_date((d, dr)), dr.end)

    def test_get_latest_date_different_digit_lengths(self):
        d1 = Date(year=9)
        d2 = Date(year=10)
        self.assertEqual(get_latest_date((d1, d2)), d2)

    def test_get_earliest_date_different_digit_lengths(self):
        d1 = Date(year=9)
        d2 = Date(year=10)
        self.assertEqual(get_earliest_date((d1, d2)), d1)


if __name__ == "__main__":
    unittest.main()
