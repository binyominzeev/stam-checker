from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch

from src.reference import (
    _flatten_text,
    _is_torah_hit,
    fetch_sefaria_text,
    resolve_reference_from_snippet,
    search_sefaria,
)


def _make_urlopen_mock(body: dict) -> MagicMock:
    raw = json.dumps(body).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class FlattenTextTests(unittest.TestCase):
    def test_plain_string(self) -> None:
        self.assertEqual(_flatten_text("אבג"), "אבג")

    def test_list_of_strings(self) -> None:
        self.assertEqual(_flatten_text(["א", "ב", "ג"]), "א ב ג")

    def test_nested_list(self) -> None:
        self.assertEqual(_flatten_text([["א", "ב"], ["ג"]]), "א ב ג")

    def test_empty_list(self) -> None:
        self.assertEqual(_flatten_text([]), "")


class IsTorahHitTests(unittest.TestCase):
    def test_torah_by_categories(self) -> None:
        hit = {"_source": {"categories": ["Tanakh", "Torah"], "path": ""}}
        self.assertTrue(_is_torah_hit(hit))

    def test_torah_by_path(self) -> None:
        hit = {"_source": {"categories": [], "path": "Tanakh/Torah/Genesis"}}
        self.assertTrue(_is_torah_hit(hit))

    def test_prophets_not_torah(self) -> None:
        hit = {
            "_source": {
                "categories": ["Tanakh", "Prophets"],
                "path": "Tanakh/Prophets/Joshua",
            }
        }
        self.assertFalse(_is_torah_hit(hit))

    def test_empty_hit(self) -> None:
        self.assertFalse(_is_torah_hit({}))


class SearchSefariaTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_returns_hits(self, mock_urlopen: MagicMock) -> None:
        hit = {
            "_source": {
                "ref": "Genesis 1:1",
                "categories": ["Tanakh", "Torah"],
                "path": "Tanakh/Torah/Genesis",
            }
        }
        mock_urlopen.return_value = _make_urlopen_mock({"hits": {"hits": [hit]}})
        hits = search_sefaria("בראשית ברא")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["_source"]["ref"], "Genesis 1:1")

    @patch("urllib.request.urlopen")
    def test_empty_results(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_mock({"hits": {"hits": []}})
        hits = search_sefaria("xyz")
        self.assertEqual(hits, [])

    @patch("urllib.request.urlopen")
    def test_raises_on_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            search_sefaria("בראשית")
        self.assertIn("Sefaria search request failed", str(ctx.exception))


class FetchSefariaTextTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_returns_text(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_mock(
            {"versions": [{"text": "בְּרֵאשִׁית בָּרָא"}]}
        )
        result = fetch_sefaria_text("Genesis 1:1")
        self.assertEqual(result, "בְּרֵאשִׁית בָּרָא")

    @patch("urllib.request.urlopen")
    def test_flattens_list_text(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_mock(
            {"versions": [{"text": ["פסוק א", "פסוק ב"]}]}
        )
        result = fetch_sefaria_text("Genesis 1:1-2")
        self.assertEqual(result, "פסוק א פסוק ב")

    @patch("urllib.request.urlopen")
    def test_raises_on_empty_versions(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _make_urlopen_mock({"versions": []})
        with self.assertRaises(RuntimeError) as ctx:
            fetch_sefaria_text("Genesis 1:1")
        self.assertIn("No text returned", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_raises_on_network_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = urllib.error.URLError("timeout")
        with self.assertRaises(RuntimeError) as ctx:
            fetch_sefaria_text("Genesis 1:1")
        self.assertIn("Sefaria text fetch failed", str(ctx.exception))


class ResolveReferenceFromSnippetTests(unittest.TestCase):
    @patch("src.reference.fetch_sefaria_text")
    @patch("src.reference.search_sefaria")
    def test_success_path(
        self, mock_search: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "_source": {
                    "ref": "Genesis 1:1",
                    "categories": ["Tanakh", "Torah"],
                    "path": "Tanakh/Torah/Genesis",
                }
            }
        ]
        mock_fetch.return_value = "בְּרֵאשִׁית בָּרָא אֱלֹהִים"
        ref, text = resolve_reference_from_snippet("בראשית ברא")
        self.assertEqual(ref, "Genesis 1:1")
        self.assertNotIn(" ", text)

    @patch("src.reference.search_sefaria")
    def test_raises_on_no_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = []
        with self.assertRaises(RuntimeError) as ctx:
            resolve_reference_from_snippet("בראשית")
        self.assertIn("No Sefaria search results", str(ctx.exception))

    @patch("src.reference.search_sefaria")
    def test_raises_on_no_torah_results(self, mock_search: MagicMock) -> None:
        mock_search.return_value = [
            {
                "_source": {
                    "ref": "Joshua 1:1",
                    "categories": ["Tanakh", "Prophets"],
                    "path": "Tanakh/Prophets/Joshua",
                }
            }
        ]
        with self.assertRaises(RuntimeError) as ctx:
            resolve_reference_from_snippet("ויהי")
        self.assertIn("No Torah", str(ctx.exception))

    @patch("src.reference.fetch_sefaria_text")
    @patch("src.reference.search_sefaria")
    def test_raises_on_empty_fetched_text(
        self, mock_search: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "_source": {
                    "ref": "Genesis 1:1",
                    "categories": ["Tanakh", "Torah"],
                    "path": "Tanakh/Torah/Genesis",
                }
            }
        ]
        mock_fetch.return_value = "   "
        with self.assertRaises(RuntimeError) as ctx:
            resolve_reference_from_snippet("בראשית")
        self.assertIn("empty text", str(ctx.exception))

    @patch("src.reference.fetch_sefaria_text")
    @patch("src.reference.search_sefaria")
    def test_uses_top_torah_result(
        self, mock_search: MagicMock, mock_fetch: MagicMock
    ) -> None:
        mock_search.return_value = [
            {
                "_source": {
                    "ref": "Joshua 1:1",
                    "categories": ["Tanakh", "Prophets"],
                    "path": "Tanakh/Prophets/Joshua",
                }
            },
            {
                "_source": {
                    "ref": "Genesis 1:1",
                    "categories": ["Tanakh", "Torah"],
                    "path": "Tanakh/Torah/Genesis",
                }
            },
        ]
        mock_fetch.return_value = "בְּרֵאשִׁית"
        ref, _ = resolve_reference_from_snippet("בראשית")
        self.assertEqual(ref, "Genesis 1:1")


if __name__ == "__main__":
    unittest.main()
