import unittest
from copy import deepcopy
from dataclasses import asdict, replace

from helpers import make_box, make_result, make_transcription

from scripts.verification_benchmark import compare, prepare
from uma_bct.domain import BoxResult


class VerificationBenchmarkTest(unittest.TestCase):
    def setUp(self):
        box = make_box()
        self.first = make_transcription(box)
        result = make_result(BoxResult(box, self.first))
        self.manifest = {"images": [result.to_dict()]}

    def test_drafts_are_not_accepted_as_ground_truth(self):
        draft = prepare(self.manifest)
        self.assertEqual(draft["images"][0]["review_status"], "unreviewed")
        with self.assertRaisesRegex(ValueError, "human-reviewed"):
            compare(draft, self.manifest)

    def test_comparison_counts_regressions_and_missing_physical_boxes(self):
        reference = prepare(self.manifest)
        reference["images"][0]["review_status"] = "reviewed"
        candidate = deepcopy(self.manifest)
        box = candidate["images"][0]["boxes"][0]
        changed = replace(self.first, unit_number="5", typed_lines=("1960.0009", "Unit 5"))
        box.update(changed.to_dict())
        box["verification"] = {"original": [asdict(self.first)]}
        report = compare(reference, candidate)
        self.assertEqual(report["newly_incorrect_fields"]["Unit Number"], 1)
        self.assertEqual(report["first_pass_exact_matches"]["Unit Number"], 1)
        self.assertEqual(report["final_exact_matches"]["Unit Number"], 0)
        candidate["images"][0]["boxes"] = []
        self.assertEqual(len(compare(reference, candidate)["missing_boxes"]), 1)
