from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils import CharacterBox


@dataclass(frozen=True)
class GapRegion:
    box: CharacterBox
    gap: int


def detect_lines(binary: np.ndarray) -> list[tuple[int, int]]:
    projection = np.sum(binary > 0, axis=1)
    threshold = max(5, int(binary.shape[1] * 0.01))
    active_rows = projection > threshold
    lines: list[tuple[int, int]] = []
    start: int | None = None

    for index, is_active in enumerate(active_rows):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= 6:
                lines.append((max(0, start - 2), min(binary.shape[0], index + 2)))
            start = None

    if start is not None and binary.shape[0] - start >= 6:
        lines.append((max(0, start - 2), binary.shape[0]))

    return lines or [(0, binary.shape[0])]


def segment_characters(binary: np.ndarray, lines: list[tuple[int, int]]) -> tuple[list[CharacterBox], list[GapRegion]]:
    boxes: list[CharacterBox] = []
    gaps: list[GapRegion] = []

    for line_index, (top, bottom) in enumerate(lines):
        line_image = binary[top:bottom, :]
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)
        line_boxes: list[CharacterBox] = []

        for component_index in range(1, component_count):
            x, y, w, h, area = stats[component_index]
            if area < 20 or w < 2 or h < 6:
                continue
            line_boxes.append(CharacterBox(int(x), int(top + y), int(w), int(h), line_index))

        line_boxes.sort(key=lambda box: box.x)
        boxes.extend(line_boxes)
        gaps.extend(_detect_gaps(line_boxes))

    return boxes, gaps


def extract_normalized_crop(binary: np.ndarray, box: CharacterBox, size: int = 32) -> np.ndarray:
    crop = binary[box.y : box.y2, box.x : box.x2]
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return resized


def _detect_gaps(line_boxes: list[CharacterBox]) -> list[GapRegion]:
    if len(line_boxes) < 2:
        return []

    raw_gaps = [current.x - previous.x2 for previous, current in zip(line_boxes, line_boxes[1:])]
    positive_gaps = [gap for gap in raw_gaps if gap > 0]
    if not positive_gaps:
        return []

    median_gap = float(np.median(np.array(positive_gaps)))
    if median_gap <= 0:
        return []

    gap_regions: list[GapRegion] = []
    for previous, current, gap in zip(line_boxes, line_boxes[1:], raw_gaps):
        if gap <= median_gap * 1.8:
            continue
        gap_height = max(previous.h, current.h)
        gap_y = min(previous.y, current.y)
        gap_box = CharacterBox(previous.x2, gap_y, gap, gap_height, previous.line_index)
        gap_regions.append(GapRegion(gap_box, gap))
    return gap_regions
