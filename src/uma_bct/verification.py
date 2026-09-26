"""Bounded, auditable second readings of existing crops.

A third reading may corroborate a candidate, but never authorizes a novel
answer or an uncertain deletion. Existing catalogue validation remains final.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .atomic import open_stable_file
from .catalogue import map_catalogue_record
from .constants import AMBIGUOUS_IDENTIFIER_FLAG, INVALID_UNIT_FLAG, TRANSCRIPTION_MISMATCH_FLAG
from .domain import BoxTranscription
from .identity import normalise_unit_claim
from .pipeline import ImageResult, warning_text
from .run_artifacts import write_json_atomic

SETTINGS_NAME = ".uma-bct-settings.json"
MODES = ("ask", "always", "never")
FIELDS = ("typed_lines", "record_identifiers", "unit_number", "handwritten_lines", "auxiliary_annotations")
GROUPS = (FIELDS[:3], (FIELDS[3],), (FIELDS[4],))


def configured_mode(arguments: argparse.Namespace) -> str:
    value = getattr(arguments, "verification", None) or os.getenv("UMA_BCT_VERIFICATION")
    root = getattr(arguments, "results_directory", None)
    if value is None and root is not None:
        path = Path(root) / SETTINGS_NAME
        try:
            with open_stable_file(path, "Verification settings") as (source, metadata):
                if metadata.st_size > 4096:
                    raise ValueError("Verification settings are too large.")
                value = json.loads(source.read())["verification"]
        except FileNotFoundError:
            pass
    value = value or "ask"
    if value not in MODES:
        raise ValueError("Verification must be ask, always, or never.")
    return value


def choose_verification(arguments: argparse.Namespace, mode: str, *, has_records: bool) -> bool:
    if not has_records or mode == "never":
        return False
    if mode == "always":
        return True
    if getattr(arguments, "json", False) or not sys.stdin.isatty() or not sys.stderr.isatty():
        return False
    print("\nFirst pass finished. Check each crop again? This adds model processing time.", file=sys.stderr)
    managed = getattr(arguments, "results_directory", None)
    choices = "[y] once, [a] always, [N] finish" if managed else "[y] once, [N] finish"
    while True:
        print(f"Verify transcriptions? {choices}: ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline().strip().casefold()
        if answer in {"", "n", "no"}:
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"a", "always"} and managed:
            write_json_atomic(Path(managed) / SETTINGS_NAME, {"verification": "always"})
            return True


def reading(value: dict[str, Any]) -> BoxTranscription:
    data = dict(value)
    for key in ("typed_lines", "record_identifiers", "handwritten_lines", "auxiliary_annotations"):
        data[key] = tuple(data[key])
    return BoxTranscription(**data)


def normalized(value: object) -> str:
    if isinstance(value, (tuple, list)):
        value = " ".join(value)
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def signature(value: BoxTranscription, fields: tuple[str, ...] = FIELDS) -> tuple[str, ...]:
    return tuple((normalise_unit_claim(value.unit_number) or normalized(value.unit_number))
                 if key == "unit_number" else normalized(getattr(value, key)) for key in fields)


def align(original: list[BoxTranscription], candidates: list[BoxTranscription]) -> list[BoxTranscription] | None:
    if len(original) != len(candidates):
        return None
    if len(original) == 1:
        return candidates
    # Text and unit numbers are precisely what may be wrong. They must not
    # be used to guess which physical box a different reading belongs to.
    positions = [normalized(item.position).casefold() for item in original]
    by_position = {normalized(item.position).casefold(): item for item in candidates}
    if len(set(positions)) != len(original) or len(by_position) != len(candidates) or not all(positions):
        return None
    if set(positions) != set(by_position):
        return None
    return [by_position[position] for position in positions]


def reconcile(first: BoxTranscription, second: BoxTranscription,
              third: BoxTranscription | None) -> tuple[BoxTranscription, list[str], list[str]]:
    selected: dict[str, Any] = {}
    disputed: list[str] = []
    unresolved: list[str] = []
    contributors = [first, second]
    for fields in GROUPS:
        a, b = signature(first, fields), signature(second, fields)
        if a == b:
            continue
        disputed.extend(fields)
        c = signature(third, fields) if third else None
        if c == a:
            contributors.append(third)
        elif c == b and not any(old and not new for old, new in zip(a, b, strict=True)):
            candidate = replace(first, **{key: getattr(second, key) for key in fields}, needs_review=False)
            flags = map_catalogue_record(candidate, "crop.png")["Review Flags"]
            if fields == GROUPS[0] and any(flag in flags for flag in (
                AMBIGUOUS_IDENTIFIER_FLAG, INVALID_UNIT_FLAG, TRANSCRIPTION_MISMATCH_FLAG
            )):
                unresolved.extend(fields)
            else:
                selected.update({key: getattr(second, key) for key in fields})
                contributors.append(third)
        else:
            unresolved.extend(fields)
    return replace(first, **selected, confidence=min(item.confidence for item in contributors),
                   needs_review=bool(disputed) or any(item.needs_review for item in contributors)), disputed, unresolved


class VerificationCheckpointError(RuntimeError):
    """A disk failure must stop verification rather than become a model warning."""


def verify_image(result: ImageResult, transcriber: Any, *, on_stage: Callable[[str], None],
                 checkpoint: Callable[[ImageResult], None] = lambda _result: None) -> ImageResult:
    records = list(result.records)
    groups: dict[Path, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.box.image_file_path].append(index)

    def save() -> None:
        status = "partial" if any(r.verification and r.verification.get("status") == "failed" for r in records) else result.status
        try:
            checkpoint(replace(result, records=tuple(records), status=status))
        except Exception as error:
            raise VerificationCheckpointError(str(error)) from error

    failed = False
    for number, indices in enumerate(groups.values(), 1):
        representative = records[indices[0]]
        audit = representative.verification
        if audit and audit.get("version", 1) != 1:
            raise ValueError("Unsupported verification evidence version.")
        if audit and audit.get("status") in {"complete", "failed"}:
            failed |= audit["status"] == "failed"
            continue
        if audit is None:
            audit = {"version": 1, "status": "pending", "original": [asdict(records[i].transcription) for i in indices]}
        for i in indices:
            records[i] = replace(records[i], verification=audit)
        save()
        first = [reading(value) for value in audit["original"]]
        box = representative.box
        full_image = len(indices) > 1 or "full-image" in box.box_id
        try:
            if "second" not in audit:
                on_stage(f"Verifying crop {number} of {len(groups)}")
                second = transcriber.verify(box, full_image=full_image)
                audit["second"] = [asdict(value) for value in second]
                save()
            second = align(first, [reading(value) for value in audit["second"]])
            if second is None:
                audit.update(disputed=list(FIELDS), unresolved=["physical_box_match"], status="complete")
                for i in indices:
                    records[i] = replace(records[i], transcription=replace(records[i].transcription, needs_review=True))
                save()
                continue
            differs = any(signature(a) != signature(b) for a, b in zip(first, second, strict=True))
            if differs and "third" not in audit:
                audit["disputed"] = list(FIELDS)
                on_stage(f"Checking differences in crop {number} of {len(groups)}")
                third = transcriber.resolve_discrepancy(box, first, second, full_image=full_image)
                audit["third"] = [asdict(value) for value in third]
                save()
            third = align(first, [reading(value) for value in audit.get("third", [])])
            disputed_fields, unresolved_fields = set(), set()
            for offset, i in enumerate(indices):
                chosen, disputed, unresolved = reconcile(first[offset], second[offset], third[offset] if third else None)
                records[i] = replace(records[i], transcription=replace(chosen, box_id=records[i].box.box_id))
                disputed_fields.update(disputed)
                unresolved_fields.update(unresolved)
            audit.update(status="complete", disputed=sorted(disputed_fields), unresolved=sorted(unresolved_fields))
        except VerificationCheckpointError:
            raise
        except Exception as error:
            failed = True
            audit.update(status="failed", error=warning_text(error))
            for i in indices:
                records[i] = replace(records[i], transcription=replace(records[i].transcription, needs_review=True))
        save()
    return replace(result, records=tuple(records), status="partial" if failed and records else result.status)
