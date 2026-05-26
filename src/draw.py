from __future__ import annotations

import cv2
import numpy as np

from src.utils import DetectedIssue


def draw_issues(image: np.ndarray, issues: list[DetectedIssue]) -> np.ndarray:
    annotated = image.copy()
    for issue in issues:
        color = _issue_color(issue)
        start = (issue.box.x, issue.box.y)
        end = (issue.box.x2, issue.box.y2)
        cv2.rectangle(annotated, start, end, color, 2)
        text = _build_text(issue)
        text_origin = (issue.box.x, max(14, issue.box.y - 6))
        cv2.putText(
            annotated,
            text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def _issue_color(issue: DetectedIssue) -> tuple[int, int, int]:
    if issue.kind == "substitution_suspicious":
        return (255, 0, 0)
    if issue.kind == "extra_character":
        return (0, 165, 255)
    if issue.spacing_hint:
        return (0, 255, 255)
    return (0, 0, 255)


def _build_text(issue: DetectedIssue) -> str:
    if issue.kind == "substitution_suspicious":
        return "sub"
    if issue.kind == "extra_character":
        return "extra"
    return "miss"
