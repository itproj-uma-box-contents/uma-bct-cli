import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

from helpers import make_box, make_result, make_transcription

from uma_bct.domain import BoxResult
from uma_bct.review_export import export_review, render_review


class ReviewExportTest(unittest.TestCase):
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
