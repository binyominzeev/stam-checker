from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.segment import extract_normalized_crop
from src.utils import CharacterBox


def recognize_characters(
    binary: np.ndarray,
    boxes: list[CharacterBox],
    reference_text: str,
) -> tuple[str, list[float]]:
    if not boxes:
        return "", []

    normalized_reference = reference_text or "?"
    glyphs = [extract_normalized_crop(binary, box).astype(np.float32) / 255.0 for box in boxes]
    templates = _build_templates(glyphs, normalized_reference)

    recognized: list[str] = []
    confidences: list[float] = []
    for glyph in glyphs:
        best_letter = "?"
        best_score = -1.0
        for letter, template in templates.items():
            score = _similarity_score(glyph, template)
            if score > best_score:
                best_letter = letter
                best_score = score
        recognized.append(best_letter)
        confidences.append(max(0.0, min(1.0, best_score)))

    return "".join(recognized), confidences


def _build_templates(glyphs: list[np.ndarray], reference_text: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    reference_length = max(1, len(reference_text))
    glyph_count = len(glyphs)

    for index, glyph in enumerate(glyphs):
        reference_index = min(reference_length - 1, int(index * reference_length / max(1, glyph_count)))
        grouped[reference_text[reference_index]].append(glyph)

    return {
        letter: np.mean(np.stack(letter_glyphs, axis=0), axis=0)
        for letter, letter_glyphs in grouped.items()
    }


def _similarity_score(glyph: np.ndarray, template: np.ndarray) -> float:
    return 1.0 - float(np.mean(np.abs(glyph - template)))
