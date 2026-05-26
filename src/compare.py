from __future__ import annotations

from difflib import SequenceMatcher

import numpy as np

from src.segment import GapRegion
from src.utils import CharacterBox, DetectedIssue


def compare_sequences(
    recognized_text: str,
    reference_text: str,
    boxes: list[CharacterBox],
    image_shape: tuple[int, int, int] | tuple[int, int],
    gaps: list[GapRegion],
) -> list[DetectedIssue]:
    matcher = SequenceMatcher(a=recognized_text, b=reference_text, autojunk=False)
    issues: list[DetectedIssue] = []

    for tag, a_start, a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            pair_count = min(a_end - a_start, b_end - b_start)
            for offset in range(pair_count):
                box = boxes[a_start + offset]
                issues.append(
                    DetectedIssue(
                        kind="substitution_suspicious",
                        box=box,
                        label="substitution",
                        expected=reference_text[b_start + offset],
                        observed=recognized_text[a_start + offset],
                    )
                )
            if a_end - a_start > pair_count:
                for box in boxes[a_start + pair_count : a_end]:
                    issues.append(
                        DetectedIssue(
                            kind="extra_character",
                            box=box,
                            label="extra",
                            observed=recognized_text[boxes.index(box)],
                        )
                    )
            if b_end - b_start > pair_count:
                missing_count = b_end - b_start - pair_count
                issues.extend(
                    _build_missing_issues(
                        boxes,
                        a_start + pair_count,
                        missing_count,
                        reference_text[b_start + pair_count : b_end],
                        image_shape,
                        gaps,
                    )
                )
            continue
        if tag == "delete":
            for index in range(a_start, a_end):
                issues.append(
                    DetectedIssue(
                        kind="extra_character",
                        box=boxes[index],
                        label="extra",
                        observed=recognized_text[index],
                    )
                )
            continue
        if tag == "insert":
            issues.extend(
                _build_missing_issues(
                    boxes,
                    a_start,
                    b_end - b_start,
                    reference_text[b_start:b_end],
                    image_shape,
                    gaps,
                )
            )

    return issues


def summarize_issues(issues: list[DetectedIssue]) -> dict[str, int]:
    counts = {
        "missing_character": 0,
        "extra_character": 0,
        "substitution_suspicious": 0,
    }
    for issue in issues:
        counts[issue.kind] = counts.get(issue.kind, 0) + 1
    return counts


def _build_missing_issues(
    boxes: list[CharacterBox],
    insert_index: int,
    count: int,
    expected_text: str,
    image_shape: tuple[int, int, int] | tuple[int, int],
    gaps: list[GapRegion],
) -> list[DetectedIssue]:
    if count <= 0:
        return []

    placeholder_boxes = estimate_missing_boxes(boxes, insert_index, count, image_shape)
    spacing_hints = _spacing_hint_flags(placeholder_boxes, gaps)
    issues: list[DetectedIssue] = []
    for offset, box in enumerate(placeholder_boxes):
        expected = expected_text[offset] if offset < len(expected_text) else ""
        issues.append(
            DetectedIssue(
                kind="missing_character",
                box=box,
                label="missing",
                expected=expected,
                spacing_hint=spacing_hints[offset],
            )
        )
    return issues


def estimate_missing_boxes(
    boxes: list[CharacterBox],
    insert_index: int,
    count: int,
    image_shape: tuple[int, int, int] | tuple[int, int],
) -> list[CharacterBox]:
    if boxes:
        median_width = int(np.median([box.w for box in boxes]))
        median_height = int(np.median([box.h for box in boxes]))
    else:
        median_width = 24
        median_height = 36

    before_box = boxes[insert_index - 1] if insert_index > 0 and insert_index - 1 < len(boxes) else None
    after_box = boxes[insert_index] if insert_index < len(boxes) else None

    height_limit = image_shape[0]
    width_limit = image_shape[1]

    if before_box and after_box and before_box.line_index == after_box.line_index:
        span_start = before_box.x2
        span_end = after_box.x
        span_width = max(median_width * count, span_end - span_start)
        step = max(median_width, int(span_width / max(1, count)))
        base_y = min(before_box.y, after_box.y)
        line_index = before_box.line_index
        return [
            CharacterBox(
                x=max(0, min(width_limit - median_width, span_start + offset * step)),
                y=max(0, min(height_limit - median_height, base_y)),
                w=min(median_width, width_limit),
                h=min(max(median_height, before_box.h, after_box.h), height_limit),
                line_index=line_index,
            )
            for offset in range(count)
        ]

    anchor = before_box or after_box
    if anchor is None:
        return [CharacterBox(0, 0, median_width, median_height, 0) for _ in range(count)]

    direction = 1 if before_box else -1
    start_x = anchor.x2 if before_box else max(0, anchor.x - median_width * count)
    return [
        CharacterBox(
            x=max(0, min(width_limit - median_width, start_x + offset * median_width * direction)),
            y=max(0, min(height_limit - median_height, anchor.y)),
            w=min(median_width, width_limit),
            h=min(max(median_height, anchor.h), height_limit),
            line_index=anchor.line_index,
        )
        for offset in range(count)
    ]


def _spacing_hint_flags(placeholder_boxes: list[CharacterBox], gaps: list[GapRegion]) -> list[bool]:
    flags: list[bool] = []
    for placeholder in placeholder_boxes:
        flags.append(
            any(
                gap.box.line_index == placeholder.line_index
                and gap.box.x <= placeholder.x2
                and gap.box.x2 >= placeholder.x
                for gap in gaps
            )
        )
    return flags
