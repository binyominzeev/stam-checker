from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.compare import compare_sequences, summarize_issues
from src.draw import _build_text
from src.main import main
from src.segment import GapRegion
from src.utils import CharacterBox, DetectedIssue


class CompareTests(unittest.TestCase):
    def test_compare_sequences_reports_expected_issue_types(self) -> None:
        boxes = [
            CharacterBox(10, 10, 10, 20, 0),
            CharacterBox(30, 10, 10, 20, 0),
        ]
        gaps = [GapRegion(CharacterBox(21, 10, 8, 20, 0), 8)]

        substitution_issues = compare_sequences("אב", "אג", boxes, (80, 120, 3), gaps)
        substitution_summary = summarize_issues(substitution_issues)
        self.assertEqual(substitution_summary["substitution_suspicious"], 1)

        issues = compare_sequences("אג", "אבג", boxes, (80, 120, 3), gaps)
        summary = summarize_issues(issues)

        self.assertEqual(summary["missing_character"], 1)
        self.assertEqual(summary["extra_character"], 0)
        self.assertTrue(any(issue.spacing_hint for issue in issues if issue.kind == "missing_character"))


class CliSmokeTests(unittest.TestCase):
    def test_overlay_labels_are_ascii_safe(self) -> None:
        label = _build_text(
            DetectedIssue(
                kind="missing_character",
                box=CharacterBox(10, 10, 10, 20, 0),
                label="missing",
                expected="א",
            )
        )
        self.assertEqual(label, "miss")

    def test_cli_writes_annotated_output_and_summary(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.png"
            reference_path = temp_path / "reference.txt"
            output_path = temp_path / "annotated.png"

            image = np.full((120, 240, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (20, 30), (40, 90), (0, 0, 0), -1)
            cv2.rectangle(image, (75, 30), (95, 90), (0, 0, 0), -1)
            cv2.rectangle(image, (130, 30), (150, 90), (0, 0, 0), -1)
            cv2.imwrite(str(image_path), image)
            reference_path.write_text("אבגד", encoding="utf-8")

            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.main",
                    "--image",
                    str(image_path),
                    "--ref",
                    str(reference_path),
                    "--out",
                    str(output_path),
                    "--debug",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=repo_root,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
            self.assertIn("Suspected errors:", result.stdout)
            self.assertIn("missing_character:", result.stdout)
            self.assertTrue((temp_path / "annotated_gray.png").exists())
            self.assertTrue((temp_path / "annotated_binary.png").exists())


class CliArgValidationTests(unittest.TestCase):
    def _run_cli(self, extra_args: list[str]) -> subprocess.CompletedProcess:
        repo_root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            ["python", "-m", "src.main", "--image", "dummy.jpg", "--out", "dummy.jpg"]
            + extra_args,
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )

    def test_neither_ref_nor_auto_ref_query_exits_with_error(self) -> None:
        result = self._run_cli([])
        self.assertNotEqual(result.returncode, 0)

    def test_both_ref_and_auto_ref_query_exits_with_error(self) -> None:
        result = self._run_cli(["--ref", "file.txt", "--auto-ref-query", "בראשית"])
        self.assertNotEqual(result.returncode, 0)

    @patch("src.main.resolve_reference_from_snippet")
    def test_auto_ref_query_mode_prints_resolved_ref(
        self, mock_resolve: MagicMock
    ) -> None:
        mock_resolve.return_value = ("Genesis 1:1", "בראשיתבראאלהים")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.png"
            output_path = temp_path / "out.png"

            image = np.full((120, 240, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            saved_argv = sys.argv
            try:
                sys.argv = [
                    "src.main",
                    "--image",
                    str(image_path),
                    "--auto-ref-query",
                    "בראשית ברא",
                    "--out",
                    str(output_path),
                ]
                buf = io.StringIO()
                with redirect_stdout(buf):
                    ret = main()
            finally:
                sys.argv = saved_argv

            self.assertEqual(ret, 0)
            self.assertIn("Genesis 1:1", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
