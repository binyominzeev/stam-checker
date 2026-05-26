from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CharacterBox:
    x: int
    y: int
    w: int
    h: int
    line_index: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class DetectedIssue:
    kind: str
    box: CharacterBox
    label: str
    expected: str = ""
    observed: str = ""
    spacing_hint: bool = False


def normalize_reference_text(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def load_reference_text(path: str | Path) -> str:
    return normalize_reference_text(Path(path).read_text(encoding="utf-8"))


def ensure_parent_directory(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_debug_image(path: str | Path, image: np.ndarray) -> None:
    ensure_parent_directory(path)
    cv2.imwrite(str(path), image)
