import unittest
from pathlib import Path

from uma_bct.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from uma_bct.transcriptions import ObservedBox, deduplicate, observations_from_response


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
                record_identifiers=("1974.0056",),
                auxiliary_annotations=("Fragile",),
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
        self.assertEqual(merged.label.record_identifiers, ("1974.0056",))
        self.assertEqual(merged.auxiliary_annotations, ("Fragile",))

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

    def test_padded_and_handwritten_unpadded_units_merge_when_crops_overlap(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (220, 200)))
        second_box = self.box("crop-2", BoundingBox((20, 0), (240, 200)))
        observations = [
            ObservedBox(
                first_box,
                self.transcription(first_box, "7", (), ("Box no. 7",), 90),
                1,
            ),
            ObservedBox(
                second_box,
                self.transcription(
                    second_box,
                    "0007",
                    ("Unit 0007",),
                    (),
                    90,
                ),
                2,
            ),
        ]

        result = deduplicate(observations)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].transcription.label.unit_number, "0007")

    def test_same_unit_on_separate_physical_boxes_is_not_merged(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (180, 200)))
        second_box = self.box("crop-2", BoundingBox((220, 0), (400, 200)))
        observations = [
            ObservedBox(
                first_box,
                self.transcription(first_box, "0001", ("Unit 0001",), (), 90),
                1,
            ),
            ObservedBox(
                second_box,
                self.transcription(second_box, "0001", ("Unit 0001",), (), 90),
                2,
            ),
        ]

        self.assertEqual(len(deduplicate(observations)), 2)

    def test_adjacent_overlapping_boxes_with_generic_typed_text_are_not_merged(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (130, 200)))
        second_box = self.box("crop-2", BoundingBox((100, 0), (230, 200)))
        observations = [
            ObservedBox(
                first_box,
                self.transcription(
                    first_box,
                    "",
                    ("Australian Pulp and Paper Mills",),
                    (),
                    90,
                ),
                1,
            ),
            ObservedBox(
                second_box,
                self.transcription(
                    second_box,
                    "",
                    ("Australian Pulp and Paper Mills",),
                    (),
                    90,
                ),
                2,
            ),
        ]

        self.assertEqual(len(deduplicate(observations)), 2)

    def test_singleton_preserves_existing_review_state(self):
        box = self.box("crop-1", BoundingBox((0, 0), (200, 200)))
        observation = ObservedBox(
            box,
            self.transcription(
                box,
                "0001",
                ("Unit 0001",),
                (),
                90,
                needs_review=True,
            ),
            1,
        )

        result = deduplicate([observation])

        self.assertTrue(result[0].transcription.needs_review)

    def test_merged_observations_preserve_any_existing_review_state(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (220, 200)))
        second_box = self.box("crop-2", BoundingBox((20, 0), (240, 200)))
        observations = [
            ObservedBox(
                first_box,
                self.transcription(
                    first_box,
                    "0001",
                    ("Unit 0001",),
                    (),
                    90,
                    needs_review=True,
                ),
                1,
            ),
            ObservedBox(
                second_box,
                self.transcription(
                    second_box,
                    "0001",
                    ("Unit 0001",),
                    (),
                    90,
                ),
                2,
            ),
        ]

        result = deduplicate(observations)

        self.assertEqual(len(result), 1)
        self.assertTrue(result[0].transcription.needs_review)

    def test_separate_results_from_one_response_are_never_merged(self):
        box = self.box("full-image", BoundingBox((0, 0), (400, 200)))
        response = [
            self.transcription(box, "0001", ("Unit 0001",), (), 90),
            self.transcription(box, "0001", ("Unit 0001",), (), 88),
        ]

        observations = observations_from_response(box, response, crop_index=0)

        self.assertEqual(len(deduplicate(observations)), 2)

    def test_other_crops_cannot_bridge_same_response_records_into_one_group(self):
        first_box = self.box("crop-1", BoundingBox((0, 0), (220, 200)))
        second_box = self.box("crop-2", BoundingBox((10, 0), (230, 200)))
        first = ObservedBox(
            first_box,
            self.transcription(first_box, "0001", ("Unit 0001",), (), 90),
            1,
        )
        second_response = [
            self.transcription(second_box, "0001", ("Unit 0001",), (), 90),
            self.transcription(second_box, "0001", ("Unit 0001",), (), 88),
        ]
        observations = [
            first,
            *observations_from_response(second_box, second_response, crop_index=2),
        ]

        result = deduplicate(observations)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            [item.transcription.label.unit_number for item in result],
            ["0001", "0001"],
        )

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
        *,
        record_identifiers: tuple[str, ...] = (),
        auxiliary_annotations: tuple[str, ...] = (),
        needs_review: bool = False,
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
            label=Label(box.box_id, unit, typed, record_identifiers),
            text_lines=lines,
            confidence=confidence,
            needs_review=needs_review,
            auxiliary_annotations=auxiliary_annotations,
        )


if __name__ == "__main__":
    unittest.main()
