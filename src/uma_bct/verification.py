"""Bounded, auditable second readings of existing crops.

A third reading may corroborate a candidate, but never authorizes a novel
answer or an uncertain deletion. Existing catalogue validation remains final.
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .catalogue import map_catalogue_record
from .constants import AMBIGUOUS_IDENTIFIER_FLAG, INVALID_UNIT_FLAG, TRANSCRIPTION_MISMATCH_FLAG
from .domain import BoxTranscription
from .identity import collect_unit_evidence, identifier_evidence_details, normalise_unit_claim
from .pipeline import ImageResult, warning_text
from .ports import DateDetector, extract_dates_with_review
from .util.date_util import deserialize_dates

MODES = ("ask", "always", "never")
FIELDS = ("typed_lines", "record_identifiers", "unit_number", "handwritten_lines", "auxiliary_annotations")
GROUPS = (FIELDS[:3], (FIELDS[3],), (FIELDS[4],))


def configured_mode(arguments: argparse.Namespace) -> str:
    value = getattr(arguments, "verification", None) or "never"
    if value not in MODES:
        raise ValueError("Verification must be ask, always, or never.")
    return value


def choose_verification(arguments: argparse.Namespace, mode: str, *, has_records: bool) -> bool:
    if not has_records or mode == "never":
        return False
    if mode == "always":
        return True
    # Keep the launcher and verification prompt rules consistent when any
    # stream is redirected for automation or captured reports.
    if getattr(arguments, "json", False) or not all(stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr)):
        return False
    print("\nFirst pass finished. Check each crop again? This adds model processing time.", file=sys.stderr)
    choices = "[y] once, [N] finish"
    while True:
        print(f"Verify transcriptions? {choices}: ", end="", file=sys.stderr, flush=True)
        answer = sys.stdin.readline().strip().casefold()
        if answer in {"", "n", "no"}:
            return False
        if answer in {"y", "yes"}:
            return True


def reading(value: dict[str, Any]) -> BoxTranscription:
    data = dict(value)
    for key in ("typed_lines", "record_identifiers", "handwritten_lines", "auxiliary_annotations"):
        data[key] = tuple(data[key])
    data["dates"] = deserialize_dates(data.get("dates", []))
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


def omits_content(original: str, candidate: str) -> bool:
    """Reject missing tokens, including deletions mixed with other corrections."""
    old = re.findall(r"\w+|[^\w\s]", original)
    new = re.findall(r"\w+|[^\w\s]", candidate)
    return any(tag == "delete" or (tag == "replace" and end_old - start_old > end_new - start_new)
               for tag, start_old, end_old, start_new, end_new
               in SequenceMatcher(None, old, new, autojunk=False).get_opcodes())


def handwritten_identity(value: BoxTranscription) -> bool:
    """Use the catalogue's evidence rules to find dependent handwriting."""
    units = collect_unit_evidence((), value.handwritten_lines)
    identifiers, malformed = identifier_evidence_details(value.handwritten_lines)
    return bool(units.candidates or units.ambiguous or identifiers or malformed)


def reconcile(first: BoxTranscription, second: BoxTranscription,
              third: BoxTranscription | None) -> tuple[BoxTranscription, list[str], list[str]]:
    selected: dict[str, Any] = {}
    disputed: list[str] = []
    unresolved: list[str] = []
    contributors = [first, second]
    groups = ((FIELDS[:4], GROUPS[2]) if any(handwritten_identity(value) for value in (first, second, third)
                                           if value is not None) else GROUPS)
    identity_flags = (AMBIGUOUS_IDENTIFIER_FLAG, INVALID_UNIT_FLAG, TRANSCRIPTION_MISMATCH_FLAG)
    for fields in groups:
        a, b = signature(first, fields), signature(second, fields)
        if a == b:
            continue
        disputed.extend(fields)
        c = signature(third, fields) if third else None
        if c == a:
            contributors.append(third)
        elif c == b and not any(omits_content(old, new) for old, new in zip(a, b, strict=True)):
            candidate = replace(first, **{key: getattr(second, key) for key in fields}, needs_review=False)
            flags = map_catalogue_record(candidate, "crop.png")["Review Flags"]
            if fields[0] == "typed_lines" and any(flag in flags for flag in identity_flags):
                unresolved.extend(fields)
            else:
                selected.update({key: getattr(second, key) for key in fields})
                contributors.append(third)
        else:
            unresolved.extend(fields)
    # Check the assembled evidence as well as each proposed group. Never turn
    # a valid original identity into a contradictory hybrid of the readings.
    original_row = map_catalogue_record(replace(first, needs_review=False), "crop.png")
    final_row = map_catalogue_record(replace(first, **selected, needs_review=False), "crop.png")
    if ((original_row["Unit Number"] and not final_row["Unit Number"])
            or any(flag in final_row["Review Flags"] and flag not in original_row["Review Flags"]
                   for flag in identity_flags)):
        for key in FIELDS[:4]:
            if key in selected:
                del selected[key]
                if key not in unresolved:
                    unresolved.append(key)
    if "handwritten_lines" in selected:
        # Dates and their failure flag describe the selected handwriting.
        # Never carry an earlier extraction failure onto a corroborated correction.
        selected["dates"] = second.dates
        selected["date_extraction_failed"] = second.date_extraction_failed
        selected["date_needs_review"] = second.date_needs_review
    else:
        # New agreement may flag old saved dates without rewriting their values.
        same_handwriting = signature(first, ("handwritten_lines",))
        selected["date_needs_review"] = first.date_needs_review or any(
            value.date_needs_review and signature(value, ("handwritten_lines",)) == same_handwriting
            for value in (second, third) if value is not None
        )
    return replace(first, **selected, confidence=min(item.confidence for item in contributors),
                   needs_review=bool(disputed) or any(item.needs_review for item in contributors)), disputed, unresolved


class VerificationCheckpointError(RuntimeError):
    """A disk failure must stop verification rather than become a model warning."""


class VerificationSetupError(RuntimeError):
    """Model setup or identity validation failed before a reading request."""


def verify_image(result: ImageResult, transcriber: Any, *, on_stage: Callable[[str], None],
                 checkpoint: Callable[[ImageResult], None] = lambda _result: None,
                 get_transcriber: Callable[[], Any] | None = None,
                 date_detector: DateDetector | None = None,
                 get_date_detector: Callable[[], DateDetector | None] | None = None) -> ImageResult:
    records = list(result.records)
    groups: dict[Path, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.box.image_file_path].append(index)

    def model() -> Any:
        nonlocal transcriber
        if transcriber is None:
            try:
                if get_transcriber is None:
                    raise ValueError("Verification requires a transcriber for missing readings.")
                transcriber = get_transcriber()
                if transcriber is None:
                    raise ValueError("Verification transcriber factory returned no transcriber.")
            except Exception as error:
                raise VerificationSetupError(str(error)) from error
        return transcriber

    def save() -> None:
        status = "partial" if any(r.verification and r.verification.get("status") == "failed" for r in records) else result.status
        try:
            checkpoint(replace(result, records=tuple(records), status=status))
        except Exception as error:
            raise VerificationCheckpointError(str(error)) from error

    def extract_dates(values: list[BoxTranscription]) -> list[BoxTranscription]:
        try:
            detector = get_date_detector() if get_date_detector is not None else date_detector
        except Exception as error:
            raise VerificationSetupError(str(error)) from error
        if detector is None:
            return values
        extracted = []
        for value in values:
            outcome = extract_dates_with_review(detector, value.get_date_text())
            extracted.append(replace(value, dates=outcome.dates, date_needs_review=outcome.needs_review))
        return extracted

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
                second = model().verify(box, full_image=full_image)
                audit["second"] = [asdict(value) for value in second]
                audit["second_dates_complete"] = False
                save()
            if audit.get("second_dates_complete") is False:
                second = extract_dates([reading(value) for value in audit["second"]])
                audit["second"] = [asdict(value) for value in second]
                audit["second_dates_complete"] = True
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
                third = model().resolve_discrepancy(box, first, second, full_image=full_image)
                audit["third"] = [asdict(value) for value in third]
                audit["third_dates_complete"] = False
                save()
            if audit.get("third_dates_complete") is False:
                third = extract_dates([reading(value) for value in audit["third"]])
                audit["third"] = [asdict(value) for value in third]
                audit["third_dates_complete"] = True
                save()
            third = align(first, [reading(value) for value in audit.get("third", [])])
            disputed_fields, unresolved_fields = set(), set()
            for offset, i in enumerate(indices):
                chosen, disputed, unresolved = reconcile(first[offset], second[offset], third[offset] if third else None)
                records[i] = replace(records[i], transcription=replace(chosen, box_id=records[i].box.box_id))
                disputed_fields.update(disputed)
                unresolved_fields.update(unresolved)
            audit.update(status="complete", disputed=sorted(disputed_fields), unresolved=sorted(unresolved_fields))
        except (VerificationCheckpointError, VerificationSetupError):
            raise
        except Exception as error:
            failed = True
            audit.update(status="failed", error=warning_text(error))
            for i in indices:
                records[i] = replace(records[i], transcription=replace(records[i].transcription, needs_review=True))
        save()
    return replace(result, records=tuple(records), status="partial" if failed and records else result.status)
