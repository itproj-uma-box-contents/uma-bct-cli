import unittest
from pathlib import Path

from src.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from src.transcriptions import ObservedBox, deduplicate, observations_from_response


class TranscriptionMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = InputImage(Path("shelf.jpg"), width=400, height=200)

    def test_different_units_from_one_wide_crop_remain_separate(self):
        box = self.box("crop-1", BoundingBox((0, 0), (200, 200)))
        response = [
            self.transcription(box, "0001", ("Unit 0001",), (), 90),
            self.transcription(box, "0002", ("Unit 0002",), (), 88),
        ]

        observations = observations_from_response(box, response, crop_index=1)
        result = deduplicate(observations)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [item.transcription.label.unit_number for item in result],
            ["0001", "0002"],
        )
        self.assertNotEqual(result[0].box.box_id, result[1].box.box_id)

    def test_duplicate_crops_merge_the_most_detailed_fields(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (220, 200)))
        second_box = self.box("crop-2", BoundingBox((20, 0), (240, 200)))
        first = ObservedBox(
            first_box,
            self.transcription(first_box, "0001", ("Unit 0001",), ("A",), 82),
            1,
        )
        second = ObservedBox(
            second_box,
            self.transcription(
                second_box,
                "0001",
                ("Collection 56", "Unit 0001"),
                ("Detailed handwriting", "Second line"),
                67,
            ),
            2,
        )

        result = deduplicate([first, second])

        self.assertEqual(len(result), 1)
        merged = result[0].transcription
        self.assertEqual(merged.typed_lines, ("Collection 56", "Unit 0001"))
        self.assertEqual(
            merged.handwritten_lines,
            ("Detailed handwriting", "Second line"),
        )
        self.assertEqual(merged.confidence, 67)
        self.assertTrue(merged.needs_review)

    def test_conflicting_units_are_never_merged(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (220, 200)))
        second_box = self.box("crop-2", BoundingBox((20, 0), (240, 200)))
        observations = [
            ObservedBox(
                first_box,
                self.transcription(first_box, "0001", ("Unit 0001",), (), 90),
                1,
            ),
            ObservedBox(
                second_box,
                self.transcription(second_box, "0002", ("Unit 0002",), (), 90),
                2,
            ),
        ]

        self.assertEqual(len(deduplicate(observations)), 2)

    def box(self, box_id: str, bounding_box: BoundingBox) -> Box:
        return Box(
            box_id=box_id,
            source_image=self.image,
            bounding_box=bounding_box,
            image_file_path=Path(f"{box_id}.png"),
            position="top-left",
            detector_confidence=0.9,
        )

    @staticmethod
    def transcription(
        box: Box,
        unit: str,
        typed: tuple[str, ...],
        handwritten: tuple[str, ...],
        confidence: int,
    ) -> BoxTranscription:
        lines = tuple(
            BoxText(index, text, "typed", confidence)
            for index, text in enumerate(typed, start=1)
        ) + tuple(
            BoxText(index, text, "handwritten", confidence)
            for index, text in enumerate(handwritten, start=1)
        )
        return BoxTranscription(
            box_id=box.box_id,
            position=box.position,
            label=Label(box.box_id, unit, typed),
            text_lines=lines,
            confidence=confidence,
        )


if __name__ == "__main__":
    unittest.main()
