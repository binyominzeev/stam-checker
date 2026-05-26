from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.utils import normalize_reference_text

_SEFARIA_SEARCH_URL = "https://www.sefaria.org/api/search-wrapper"
_SEFARIA_TEXTS_URL = "https://www.sefaria.org/api/v3/texts/{ref}"


def load_reference_from_file(path: str | Path) -> str:
    """Load and normalize a local plain-text Hebrew reference file."""
    return normalize_reference_text(Path(path).read_text(encoding="utf-8"))


def search_sefaria(snippet: str) -> list[dict]:
    """Call the Sefaria search-wrapper API and return the raw list of hits."""
    payload = json.dumps(
        {
            "query": snippet,
            "type": "text",
            "filters": [],
            "field": "naive_lemmatizer",
            "slop": 3,
            "sort_type": "score",
            "sort_direction": "desc",
            "size": 10,
            "from": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        _SEFARIA_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Sefaria search request failed: {exc}") from exc
    hits = data.get("hits", {}).get("hits", [])
    if not isinstance(hits, list):
        raise RuntimeError("Unexpected Sefaria search response format.")
    return hits


def _is_torah_hit(hit: dict) -> bool:
    """Return True if the search hit belongs to Tanakh > Torah."""
    source = hit.get("_source", {})
    categories: list = source.get("categories", [])
    if "Tanakh" in categories and "Torah" in categories:
        return True
    path: str = source.get("path", "")
    return "Torah" in path.split("/")


def fetch_sefaria_text(ref: str) -> str:
    """Fetch canonical Hebrew text for a Sefaria reference string.

    Uses the Sefaria v3 Texts API with the *Tanach with Text Only* Hebrew
    version, which returns unpointed text without nikud or cantillation marks.
    """
    encoded_ref = urllib.parse.quote(ref, safe="")
    url = _SEFARIA_TEXTS_URL.format(ref=encoded_ref)
    url += "?version=hebrew%7CTanach_with_Text_Only"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Sefaria text fetch failed for '{ref}': {exc}") from exc
    versions = data.get("versions", [])
    if not versions:
        raise RuntimeError(f"No text returned by Sefaria for ref '{ref}'.")
    text = versions[0].get("text", "")
    return _flatten_text(text)


def _flatten_text(text: object) -> str:
    """Recursively flatten nested text arrays into a single whitespace-joined string."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return " ".join(_flatten_text(item) for item in text)
    return ""


def resolve_reference_from_snippet(snippet: str) -> tuple[str, str]:
    """Resolve a rough Hebrew snippet to a canonical reference and normalised text.

    Uses Sefaria search to identify the source passage, then fetches its
    canonical Hebrew text.

    Returns:
        A ``(ref, normalised_text)`` pair where *ref* is the Sefaria reference
        string (e.g. ``"Genesis 1:1"``) and *normalised_text* is the whitespace-
        stripped Hebrew text suitable for the existing comparison pipeline.

    Raises:
        RuntimeError: on no search results, no Torah result, fetch failure, or
            empty fetched text.
    """
    hits = search_sefaria(snippet)
    if not hits:
        raise RuntimeError("No Sefaria search results found for the given snippet.")

    torah_hits = [h for h in hits if _is_torah_hit(h)]
    if not torah_hits:
        raise RuntimeError(
            "No Torah (Tanakh > Torah) results found in Sefaria search results."
        )

    ref = torah_hits[0].get("_source", {}).get("ref", "")
    if not ref:
        raise RuntimeError("Sefaria search result is missing the 'ref' field.")

    raw_text = fetch_sefaria_text(ref)
    if not raw_text.strip():
        raise RuntimeError(f"Sefaria returned empty text for ref '{ref}'.")

    return ref, normalize_reference_text(raw_text)
