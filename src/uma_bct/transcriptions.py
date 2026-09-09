"""Combine overlapping crop responses without merging physical boxes.

Detectors intentionally over-propose padded crops, so the same physical box
can be photographed (and transcribed) more than once. ``deduplicate`` groups
``ObservedBox`` results using ``is_duplicate``, which requires both geometric
overlap (crop bounding-box IoU) and content agreement (matching unit
evidence, near-identical typed text, or a shared position hint) -- and
refuses to merge across crops with conflicting unit numbers even when they
overlap. ``merge_group`` then keeps, per field independently, whichever
observation has the most detail (longest non-empty text, then confidence)
rather than the first or "best overall" observation, and forces
``needs_review`` whenever the merged observations actually disagreed with
each other.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from .domain import Box, BoxTranscription
from .identity import CONTEXTUAL_UNIT, normalise_unit_claim
from .imaging import iou

TYPED_SIMILARITY_THRESHOLD = 0.92
OVERLAP_THRESHOLD = 0.70


@dataclass(frozen=True)
class ObservedBox:
    box: Box
    transcription: BoxTranscription
    crop_index: int


def observations_from_response(
    box: Box, transcriptions: Sequence[BoxTranscription], crop_index: int
) -> list[ObservedBox]:
    """Give every record from one model response a distinct stable ID."""

    observations: list[ObservedBox] = []
    for response_index, transcription in enumerate(transcriptions, start=1):
        box_id = (
            f"{box.box_id}:result-{response_index:03d}"
            if len(transcriptions) > 1
            else box.box_id
        )
        observations.append(
            ObservedBox(
                replace(box, box_id=box_id),
                replace(transcription, box_id=box_id),
                crop_index,
            )
        )
    return observations


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _normalise_lines(lines: Sequence[str]) -> str:
    return _normalise("\n".join(lines))


def _unit_evidence(item: ObservedBox) -> set[str]:
    evidence: set[str] = set()
    if item.transcription.unit_number.strip():
        value = item.transcription.unit_number
        evidence.add(normalise_unit_claim(value) or value.casefold())
    for line in (
        *item.transcription.typed_lines,
        *item.transcription.handwritten_lines,
    ):
        evidence.update(
            match.group(1).zfill(4) for match in CONTEXTUAL_UNIT.finditer(line)
        )
        if normalized := normalise_unit_claim(line):
            evidence.add(normalized)
    return evidence


def _unit_conflict(first: ObservedBox, second: ObservedBox) -> bool:
    first_units = _unit_evidence(first)
    second_units = _unit_evidence(second)
    return bool(first_units and second_units and first_units.isdisjoint(second_units))


def is_duplicate(first: ObservedBox, second: ObservedBox) -> bool:
    """Return whether two independently-created crops describe the same box."""

    if first.crop_index == second.crop_index or _unit_conflict(first, second):
        return False
    overlap = iou(first.box.bounding_box, second.box.bounding_box)
    if overlap < OVERLAP_THRESHOLD:
        return False

    first_units = _unit_evidence(first)
    second_units = _unit_evidence(second)
    if first_units and second_units and not first_units.isdisjoint(second_units):
        return True

    first_typed = _normalise_lines(first.transcription.typed_lines)
    second_typed = _normalise_lines(second.transcription.typed_lines)
    if first_typed and second_typed:
        return (
            SequenceMatcher(None, first_typed, second_typed).ratio()
            >= TYPED_SIMILARITY_THRESHOLD
        )
    first_position = _normalise(first.transcription.position)
    second_position = _normalise(second.transcription.position)
    return bool(first_position and first_position == second_position)


def _detail(lines: Sequence[str]) -> tuple[int, int]:
    non_empty = tuple(line for line in lines if line.strip())
    return sum(len("".join(line.split())) for line in non_empty), len(non_empty)


def _best(
    group: Sequence[ObservedBox],
    value: Callable[[BoxTranscription], Sequence[str]],
) -> ObservedBox:
    return max(
        group,
        key=lambda item: (
            _detail(value(item.transcription)),
            item.transcription.confidence,
            -item.crop_index,
        ),
    )


def _signature(item: ObservedBox) -> tuple[object, ...]:
    transcription = item.transcription
    return (
        _normalise(transcription.position),
        _normalise(transcription.unit_number),
        _normalise_lines(transcription.record_identifiers),
        _normalise_lines(transcription.typed_lines),
        _normalise_lines(transcription.handwritten_lines),
        _normalise_lines(transcription.auxiliary_annotations),
    )


def merge_group(group: Sequence[ObservedBox]) -> ObservedBox:
    """Keep the most detailed value for each field and flag disagreements."""

    representative = _best(group, lambda value: value.typed_lines)
    typed = representative.transcription.typed_lines
    handwritten_source = _best(group, lambda value: value.handwritten_lines)
    identifier_source = _best(group, lambda value: value.record_identifiers)
    auxiliary_source = _best(group, lambda value: value.auxiliary_annotations)
    unit_source = max(
        group,
        key=lambda item: (
            bool(re.fullmatch(r"\d{4}", item.transcription.unit_number.strip())),
            bool(item.transcription.unit_number.strip()),
            item.transcription.confidence,
            -item.crop_index,
        ),
    )
    position_source = max(
        group,
        key=lambda item: (
            bool(item.transcription.position.strip()),
            item.transcription.confidence,
            -item.crop_index,
        ),
    )
    selected = (
        representative,
        handwritten_source,
        identifier_source,
        auxiliary_source,
        unit_source,
        position_source,
    )
    transcription = BoxTranscription(
        box_id=representative.box.box_id,
        position=position_source.transcription.position,
        confidence=min(item.transcription.confidence for item in selected),
        record_identifiers=identifier_source.transcription.record_identifiers,
        unit_number=unit_source.transcription.unit_number,
        typed_lines=typed,
        handwritten_lines=handwritten_source.transcription.handwritten_lines,
        auxiliary_annotations=auxiliary_source.transcription.auxiliary_annotations,
        needs_review=(
            any(item.transcription.needs_review for item in group)
            or len({_signature(item) for item in group}) > 1
        ),
    )
    return ObservedBox(representative.box, transcription, representative.crop_index)


def deduplicate(observations: Sequence[ObservedBox]) -> list[ObservedBox]:
    groups: list[list[ObservedBox]] = []
    for observation in observations:
        group = next(
            (
                candidates
                for candidates in groups
                if all(
                    member.crop_index != observation.crop_index
                    and not _unit_conflict(observation, member)
                    for member in candidates
                )
                and any(is_duplicate(observation, member) for member in candidates)
            ),
            None,
        )
        if group is None:
            groups.append([observation])
        else:
            group.append(observation)
    return [merge_group(group) for group in groups]
