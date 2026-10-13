#!/usr/bin/env python3
"""Prepare private reference drafts and compare a run with human-reviewed labels.

Run locally only. Drafts are deliberately ineligible for accuracy measurement;
reviewers must check every photo, add missed boxes, and remove false detections.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from uma_bct.catalogue import map_catalogue_record
from uma_bct.domain import BoxTranscription
from uma_bct.run_artifacts import write_json_atomic
from uma_bct.verification import reading

COLUMNS = ("Unit Type", "Unit Number", "Description", "Date Range")


def source_filename(value: str) -> str:
    """Read filenames from either platform's serialized source paths.

    Native Windows paths use backslashes; POSIX paths may contain literal
    backslashes in filenames. Preserve slash-form paths before interpreting
    legacy Windows relative paths, independently of the reviewing host OS.
    """

    windows_path = PureWindowsPath(value)
    if value.startswith("/") or ("/" in value and not windows_path.drive):
        return PurePosixPath(value).name
    return windows_path.name


def transcription(box: dict[str, Any]) -> BoxTranscription:
    return BoxTranscription(
        box_id=box["box_id"], position=box.get("position", ""),
        confidence=box.get("transcription_confidence", 0),
        record_identifiers=tuple(box.get("record_identifiers", [])), unit_number=box.get("unit", ""),
        typed_lines=tuple(box.get("typed", [])), handwritten_lines=tuple(box.get("handwritten", [])),
        auxiliary_annotations=tuple(box.get("auxiliary_annotations", [])), needs_review=box.get("needs_review", False),
    )


def fields(value: BoxTranscription) -> dict[str, str]:
    row = map_catalogue_record(value, "crop.png", spreadsheet_safe=False)
    return {key: str(row[key]) for key in COLUMNS}


def prepare(manifest: dict[str, Any]) -> dict[str, Any]:
    images = []
    seen = set()
    for image in manifest.get("images", []):
        name = source_filename(image["source_image"])
        if name in seen:
            raise ValueError(f"Duplicate source filename requires separate reference sets: {name}")
        seen.add(name)
        images.append({
            "source_image": name, "source_path": image["source_image"], "review_status": "unreviewed",
            "notes": "Check all visible boxes; add missed boxes and remove false detections.",
            "boxes": [{"box_id": box["box_id"], "bbox": box.get("bbox"), "crop_image": box.get("image"),
                       "expected": fields(transcription(box))} for box in image.get("boxes", [])],
        })
    if not images:
        raise ValueError("The manifest contains no photo results to review.")
    return {"schema_version": 1, "description": "Draft reference: model output, not human ground truth.", "images": images}


def compare(reference: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    images = reference.get("images", [])
    if not images or any(image.get("review_status") != "reviewed" for image in images):
        raise ValueError("Every reference photo must be human-reviewed before measuring accuracy.")
    expected = {}
    for image in images:
        for box in image["boxes"]:
            key = (image["source_image"], box["box_id"])
            if key in expected or set(box["expected"]) != set(COLUMNS):
                raise ValueError("Reference boxes require unique IDs and all four catalogue fields.")
            expected[key] = box["expected"]
    actual = {}
    for image in manifest.get("images", []):
        for box in image.get("boxes", []):
            key = (source_filename(image["source_image"]), box["box_id"])
            if key in actual:
                raise ValueError("Run contains duplicate physical box IDs.")
            actual[key] = box
    matches, baseline_matches, regressions, changes = Counter(), Counter(), Counter(), Counter()
    for key in expected.keys() & actual.keys():
        box = actual[key]
        final = fields(transcription(box))
        original = next((value for value in box.get("verification", {}).get("original", [])
                         if value["box_id"] == box["box_id"]), None)
        baseline = fields(reading(original)) if original else final
        for field in COLUMNS:
            correct = final[field] == expected[key][field]
            was_correct = baseline[field] == expected[key][field]
            matches[field] += correct
            baseline_matches[field] += was_correct
            regressions[field] += was_correct and not correct
            changes[field] += baseline[field] != final[field]
    return {
        "reference_photos": len(images), "reference_boxes": len(expected), "run_boxes": len(actual),
        "matched_boxes": len(expected.keys() & actual.keys()),
        "missing_boxes": [list(key) for key in sorted(expected.keys() - actual.keys())],
        "unexpected_boxes": [list(key) for key in sorted(actual.keys() - expected.keys())],
        "first_pass_exact_matches": {field: baseline_matches[field] for field in COLUMNS},
        "final_exact_matches": {field: matches[field] for field in COLUMNS},
        "newly_incorrect_fields": {field: regressions[field] for field in COLUMNS},
        "changed_fields": {field: changes[field] for field in COLUMNS},
        "note": "Box IDs must still refer to the same physical boxes; review changed detection layouts before comparison.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "compare"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.action == "prepare":
        value = prepare(manifest)
    else:
        if args.reference is None:
            parser.error("compare requires --reference")
        value = compare(json.loads(args.reference.read_text(encoding="utf-8")), manifest)
    write_json_atomic(args.output, value, replace_existing=False)
    print(f"Saved local {args.action} output to {args.output}")


if __name__ == "__main__":
    main()
