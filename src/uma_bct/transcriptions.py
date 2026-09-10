"""Merge repeat readings of the same box without joining separate boxes.

Boxes need matching location and text before they can merge. Conflicting unit
numbers stay separate, and any disagreement is sent for review.
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


def observations_from_response(box: Box, transcriptions: Sequence[BoxTranscription], crop_index: int) -> list[ObservedBox]:
    """Give each returned box its own stable ID."""

    observations: list[ObservedBox] = []
    for response_index, transcription in enumerate(transcriptions, start=1):
        box_id = f"{box.box_id}:result-{response_index:03d}" if len(transcriptions) > 1 else box.box_id
        observations.append(ObservedBox(replace(box, box_id=box_id), replace(transcription, box_id=box_id), crop_index))
    return observations


def normalise_text(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def normalise_lines(lines: Sequence[str]) -> str:
    return normalise_text("\n".join(lines))


def unit_evidence(item: ObservedBox) -> set[str]:
    evidence: set[str] = set()
    if item.transcription.unit_number.strip():
        value = item.transcription.unit_number
        evidence.add(normalise_unit_claim(value) or value.casefold())
    for line in (*item.transcription.typed_lines, *item.transcription.handwritten_lines):
        evidence.update(match.group(1).zfill(4) for match in CONTEXTUAL_UNIT.finditer(line))
        if normalized := normalise_unit_claim(line):
            evidence.add(normalized)
    return evidence


def unit_conflict(first: ObservedBox, second: ObservedBox) -> bool:
    first_units = unit_evidence(first)
    second_units = unit_evidence(second)
    return bool(first_units and second_units and first_units.isdisjoint(second_units))


def is_duplicate(first: ObservedBox, second: ObservedBox) -> bool:
    """Check whether two crops show the same box."""

    if first.crop_index == second.crop_index or unit_conflict(first, second):
        return False
    overlap = iou(first.box.bounding_box, second.box.bounding_box)
    if overlap < OVERLAP_THRESHOLD:
        return False

    first_units = unit_evidence(first)
    second_units = unit_evidence(second)
    if first_units and second_units and not first_units.isdisjoint(second_units):
        return True

    first_typed = normalise_lines(first.transcription.typed_lines)
    second_typed = normalise_lines(second.transcription.typed_lines)
    if first_typed and second_typed:
        return SequenceMatcher(None, first_typed, second_typed).ratio() >= TYPED_SIMILARITY_THRESHOLD
    first_position = normalise_text(first.transcription.position)
    second_position = normalise_text(second.transcription.position)
    return bool(first_position and first_position == second_position)


def text_detail(lines: Sequence[str]) -> tuple[int, int]:
    non_empty = tuple(line for line in lines if line.strip())
    return sum(len("".join(line.split())) for line in non_empty), len(non_empty)


def best_observation(group: Sequence[ObservedBox], value: Callable[[BoxTranscription], Sequence[str]]) -> ObservedBox:
    return max(
        group, key=lambda item: (text_detail(value(item.transcription)), item.transcription.confidence, -item.crop_index)
    )


def transcription_signature(item: ObservedBox) -> tuple[object, ...]:
    transcription = item.transcription
    return (
        normalise_text(transcription.position),
        normalise_text(transcription.unit_number),
        normalise_lines(transcription.record_identifiers),
        normalise_lines(transcription.typed_lines),
        normalise_lines(transcription.handwritten_lines),
        normalise_lines(transcription.auxiliary_annotations),
    )


def merge_group(group: Sequence[ObservedBox]) -> ObservedBox:
    """Keep the best text from matching crops and flag differences."""

    representative = best_observation(group, lambda value: value.typed_lines)
    typed = representative.transcription.typed_lines
    handwritten_source = best_observation(group, lambda value: value.handwritten_lines)
    identifier_source = best_observation(group, lambda value: value.record_identifiers)
    auxiliary_source = best_observation(group, lambda value: value.auxiliary_annotations)
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
        group, key=lambda item: (bool(item.transcription.position.strip()), item.transcription.confidence, -item.crop_index)
    )
    selected = (representative, handwritten_source, identifier_source, auxiliary_source, unit_source, position_source)
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
            or len({transcription_signature(item) for item in group}) > 1
        ),
    )
    return ObservedBox(representative.box, transcription, representative.crop_index)


def deduplicate(observations: Sequence[ObservedBox]) -> list[ObservedBox]:
    groups: list[list[ObservedBox]] = []
    # Merge matching crops, but never merge two records from the same crop.
    for observation in observations:
        group = next(
            (
                candidates
                for candidates in groups
                if all(
                    member.crop_index != observation.crop_index and not unit_conflict(observation, member)
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
