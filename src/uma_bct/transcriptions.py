"""Combine multi-box transcription responses without detector coupling."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from .domain import Box, BoxText, BoxTranscription, Label
from .imaging import iou

TYPED_SIMILARITY_THRESHOLD = 0.92
TYPED_OVERLAP_THRESHOLD = 0.70
SPATIAL_ONLY_OVERLAP_THRESHOLD = 0.70


@dataclass(frozen=True)
class ObservedBox:
    """One generic transcription tied to the crop that produced it."""

    box: Box
    transcription: BoxTranscription
    crop_index: int


def with_box_id(transcription: BoxTranscription, box_id: str) -> BoxTranscription:
    label = (
        replace(transcription.label, box_id=box_id)
        if transcription.label is not None
        else None
    )
    return replace(transcription, box_id=box_id, label=label)


def observations_from_response(box: Box, transcriptions: Sequence[BoxTranscription], 
                               crop_index: int) -> list[ObservedBox]:
    """Give every record from one crop a stable, exportable identity."""

    observations: list[ObservedBox] = []
    multiple_records = len(transcriptions) > 1
    for response_index, transcription in enumerate(transcriptions, start=1):
        box_id = (
            f"{box.box_id}:result-{response_index:03d}"
            if multiple_records
            else box.box_id
        )
        observations.append(
            ObservedBox(
                box=replace(box, box_id=box_id),
                transcription=with_box_id(transcription, box_id),
                crop_index=crop_index,
            )
        )
    return observations


def normalise_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def normalise_lines(lines: Sequence[str]) -> str:
    return normalise_text("\n".join(lines))


def unit_value(item: ObservedBox) -> str:
    return item.transcription.label.unit_number if item.transcription.label else ""


def unit_evidence(item: ObservedBox) -> set[str]:
    def canonical(value: str) -> str:
        return str(int(value)).zfill(4)

    evidence: set[str] = set()
    stated_unit = unit_value(item).strip().casefold()
    if stated_unit:
        match = re.fullmatch(
            r"(?:unit\s*[:#-]?\s*)?(\d{1,4})",
            stated_unit,
            re.IGNORECASE,
        )
        evidence.add(canonical(match.group(1)) if match else stated_unit)
    for line in (
        *item.transcription.typed_lines,
        *item.transcription.handwritten_lines,
    ):
        for match in re.finditer(
            r"\b(?:unit|box(?:\s*(?:no\.?|number))?)\s*[:#-]?\s*(\d{1,4})\b",
            line,
            re.IGNORECASE,
        ):
            evidence.add(canonical(match.group(1)))
        if match := re.fullmatch(r"\s*(\d{1,4})\s*", line):
            evidence.add(canonical(match.group(1)))
    return evidence


def unit_conflict(first: ObservedBox, second: ObservedBox) -> bool:
    first_units = unit_evidence(first)
    second_units = unit_evidence(second)
    return bool(first_units and second_units and first_units.isdisjoint(second_units))


def is_duplicate(first: ObservedBox, second: ObservedBox) -> bool:
    """Return whether overlapping crop observations describe one physical box.

    Separate records from one model response are always preserved.  A matching
    unit number is useful evidence only when two independently-created crops
    strongly overlap; units can legitimately repeat on separate physical boxes.
    """

    if first.crop_index == second.crop_index:
        return False

    if unit_conflict(first, second):
        return False

    overlap = iou(first.box.bounding_box, second.box.bounding_box)
    if overlap <= 0.0:
        return False

    first_units = unit_evidence(first)
    second_units = unit_evidence(second)
    if (
        first_units
        and second_units
        and not first_units.isdisjoint(second_units)
        and overlap >= SPATIAL_ONLY_OVERLAP_THRESHOLD
    ):
        return True

    first_typed = normalise_lines(first.transcription.typed_lines)
    second_typed = normalise_lines(second.transcription.typed_lines)
    if first_typed and second_typed:
        similarity = SequenceMatcher(None, first_typed, second_typed).ratio()
        if (
            similarity >= TYPED_SIMILARITY_THRESHOLD
            and overlap >= TYPED_OVERLAP_THRESHOLD
        ):
            return True

    first_position = normalise_text(first.transcription.position)
    second_position = normalise_text(second.transcription.position)
    same_position = bool(first_position and first_position == second_position)
    return same_position and overlap >= SPATIAL_ONLY_OVERLAP_THRESHOLD


def record_signature(item: ObservedBox) -> tuple[str, str, str, str, str, str]:
    return (
        normalise_text(item.transcription.position),
        normalise_text(unit_value(item)),
        normalise_lines(item.transcription.typed_lines),
        normalise_lines(item.transcription.handwritten_lines),
        normalise_lines(
            item.transcription.label.record_identifiers
            if item.transcription.label
            else ()
        ),
        normalise_lines(item.transcription.auxiliary_annotations),
    )


def line_detail(lines: Sequence[str]) -> tuple[int, int]:
    non_empty = [line for line in lines if line.strip()]
    characters = sum(len("".join(line.split())) for line in non_empty)
    return characters, len(non_empty)


def model_tie_breaker(item: ObservedBox) -> tuple[int, int]:
    return item.transcription.confidence, -item.crop_index


def merge_group(group: list[ObservedBox]) -> ObservedBox:
    """Merge duplicate observations using the reference field-selection rules."""

    positions = [item for item in group if item.transcription.position.strip()]
    units = [item for item in group if unit_value(item).strip()]

    position_source = max(positions or group, key=model_tie_breaker)
    unit_source = max(
        units or group,
        key=lambda item: (
            bool(re.fullmatch(r"\d{4}", unit_value(item).strip())),
            *model_tie_breaker(item),
        ),
    )
    typed_source = max(
        group,
        key=lambda item: (
            line_detail(item.transcription.typed_lines),
            *model_tie_breaker(item),
        ),
    )
    handwritten_source = max(
        group,
        key=lambda item: (
            line_detail(item.transcription.handwritten_lines),
            *model_tie_breaker(item),
        ),
    )
    identifier_source = max(
        group,
        key=lambda item: (
            line_detail(
                item.transcription.label.record_identifiers
                if item.transcription.label
                else ()
            ),
            *model_tie_breaker(item),
        ),
    )
    auxiliary_source = max(
        group,
        key=lambda item: (
            line_detail(item.transcription.auxiliary_annotations),
            *model_tie_breaker(item),
        ),
    )

    confidence = min(
        source.transcription.confidence
        for source in (
            unit_source,
            typed_source,
            handwritten_source,
            identifier_source,
            auxiliary_source,
        )
    )
    unit = unit_value(unit_source)
    typed = typed_source.transcription.typed_lines
    handwritten = handwritten_source.transcription.handwritten_lines
    record_identifiers = (
        identifier_source.transcription.label.record_identifiers
        if identifier_source.transcription.label
        else ()
    )
    auxiliary_annotations = auxiliary_source.transcription.auxiliary_annotations
    representative = typed_source
    label = (
        Label(
            box_id=representative.box.box_id,
            unit_number=unit,
            typed_lines=tuple(typed),
            record_identifiers=tuple(record_identifiers),
        )
        if unit or typed or record_identifiers
        else None
    )
    lines = tuple(
        BoxText(position, text, "typed", confidence)
        for position, text in enumerate(typed, start=1)
    ) + tuple(
        BoxText(position, text, "handwritten", confidence)
        for position, text in enumerate(handwritten, start=1)
    )
    merged = BoxTranscription(
        box_id=representative.box.box_id,
        position=position_source.transcription.position,
        label=label,
        text_lines=lines,
        confidence=confidence,
        needs_review=(
            any(item.transcription.needs_review for item in group)
            or len({record_signature(item) for item in group}) > 1
        ),
        auxiliary_annotations=tuple(auxiliary_annotations),
    )
    return ObservedBox(representative.box, merged, representative.crop_index)


def deduplicate(observations: Sequence[ObservedBox]) -> list[ObservedBox]:
    """Group and merge observations using the old transcriber behaviour."""

    groups: list[list[ObservedBox]] = []
    for observation in observations:
        matching_group: list[ObservedBox] | None = None
        for group in groups:
            if any(member.crop_index == observation.crop_index for member in group):
                continue
            if any(unit_conflict(observation, member) for member in group):
                continue
            if any(is_duplicate(observation, member) for member in group):
                matching_group = group
                break
        if matching_group is None:
            groups.append([observation])
        else:
            matching_group.append(observation)
    return [merge_group(group) for group in groups]
