import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from helpers import make_box, make_result, make_transcription

from uma_bct.domain import BoxResult
from uma_bct.export import export_catalogue
from uma_bct.review_export import export_review, render_review


class ReviewExportTest(unittest.TestCase):
    def test_csv_and_html_share_default_and_explicit_confidence_thresholds(self):
        cases = (
            (49, {}, True),
            (50, {}, False),
            (79, {}, False),
            (79, {"minimum_confidence": 80}, True),
            (80, {"minimum_confidence": 80}, False),
            (0, {"minimum_confidence": 0}, False),
            (99, {"minimum_confidence": 100}, True),
            (100, {"minimum_confidence": 100}, False),
        )
        for confidence, options, flagged in cases:
            with self.subTest(confidence=confidence, options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                box = make_box(crop_path=root / "Box Images" / "box.png")
                results = [make_result(BoxResult(box, make_transcription(box, confidence=confidence)))]
                catalogue = root / "Catalogue.csv"
                review = root / "Review.html"
                export_catalogue(results, catalogue, **options)
                export_review(results, review, catalogue_path=catalogue, **options)
                with catalogue.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                self.assertEqual("Low confidence" in rows[0]["Review Flags"], flagged)
                self.assertEqual("Low confidence" in review.read_text(encoding="utf-8"), flagged)
                self.assertEqual("Low confidence" in render_review(results, catalogue, **options), flagged)

    def test_links_are_encoded_relative_paths_and_text_cannot_become_markup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / 'Box Images' / 'é # 20% "box".png'
            box = make_box(crop_path=crop)
            transcription = replace(make_transcription(box), handwritten_lines=('<script>alert(1)</script>', '=SUM(1,2)'))
            result = make_result(BoxResult(box, transcription))
            text = render_review([result], root / 'Catalogue.csv')
            self.assertIn(quote('./Box Images/' + crop.name, safe='/'), text)
            self.assertNotIn('<script>', text)
            self.assertIn('&lt;script&gt;', text)
            self.assertIn('=SUM(1,2)', text)
            self.assertNotIn("'=SUM", text)
            self.assertIn('table-layout: fixed', text)
            self.assertIn('0004', text)
            self.assertNotIn('https://', text)

    def test_export_never_clobbers_existing_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'Review.html'
            path.write_text('reviewer file')
            with self.assertRaises(FileExistsError):
                export_review([], path, catalogue_path=root / 'Catalogue.csv')
            self.assertEqual(path.read_text(), 'reviewer file')
            self.assertEqual(list(root.glob('.uma-bct-review-*')), [])

    def test_excel_review_links_to_the_workbook_and_uses_excel_wording(self):
        text = render_review([], Path("Catalogue.xlsx"))
        self.assertIn('href="./Catalogue.xlsx"', text)
        self.assertIn("Open catalogue Excel", text)
        self.assertIn("Later Excel edits", text)
        self.assertNotIn("CSV", text)
