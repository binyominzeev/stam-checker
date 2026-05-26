from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.compare import compare_sequences, summarize_issues
from src.draw import draw_issues
from src.match import recognize_characters
from src.preprocess import load_image, preprocess_image
from src.segment import detect_lines, segment_characters
from src.utils import ensure_parent_directory, load_reference_text, save_debug_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Highlight likely STaM text anomalies on an image.")
    parser.add_argument("--image", required=True, help="Path to the input JPG/PNG image.")
    parser.add_argument("--ref", required=True, help="Path to the plain-text Hebrew reference file.")
    parser.add_argument("--out", required=True, help="Path to the annotated output image.")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image = load_image(args.image)
    reference_text = load_reference_text(args.ref)
    gray, binary = preprocess_image(image)
    lines = detect_lines(binary)
    boxes, gaps = segment_characters(binary, lines)
    recognized_text, _ = recognize_characters(binary, boxes, reference_text)
    issues = compare_sequences(recognized_text, reference_text, boxes, image.shape, gaps)
    annotated = draw_issues(image, issues)

    ensure_parent_directory(args.out)
    cv2.imwrite(args.out, annotated)

    if args.debug:
        _save_debug_outputs(args.out, gray, binary)

    summary = summarize_issues(issues)
    total = sum(summary.values())
    print(f"Suspected errors: {total}")
    for key in ("missing_character", "extra_character", "substitution_suspicious"):
        print(f"{key}: {summary[key]}")

    return 0


def _save_debug_outputs(output_path: str, gray: np.ndarray, binary: np.ndarray) -> None:
    output = Path(output_path)
    save_debug_image(output.with_name(f"{output.stem}_gray.png"), gray)
    save_debug_image(output.with_name(f"{output.stem}_binary.png"), binary)


if __name__ == "__main__":
    raise SystemExit(main())
