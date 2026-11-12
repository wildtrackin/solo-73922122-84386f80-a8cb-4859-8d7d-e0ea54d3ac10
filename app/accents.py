"""Accent normalisation shared by the writer, the indexer and the query parser.

The ``keep`` policy only normalises to NFC so that a pre-composed and a
decomposed spelling of the same accented character are indexed identically.
The ``fold`` policy additionally removes combining marks that belong to Latin
base letters.  Non-Latin combining marks are deliberately preserved.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

AccentPolicy = Literal["keep", "fold"]
VALID_POLICIES: tuple[str, ...] = ("keep", "fold")

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _is_latin_base(char: str) -> bool:
    if not char.isalpha():
        return False
    if ord(char) < 128:
        return True
    return unicodedata.name(char, "").startswith("LATIN")


def fold_latin_accents(text: str) -> str:
    """Remove combining diacritics from Latin letters, preserving everything else.

    For example ``Crème brûlée`` becomes ``Creme brulee`` while a combining
    mark attached to a non-Latin base character is left untouched.
    """

    decomposed = unicodedata.normalize("NFD", text)
    output: list[str] = []
    base_is_latin = False
    for char in decomposed:
        if unicodedata.combining(char):
            if not base_is_latin:
                output.append(char)
            continue
        output.append(char)
        base_is_latin = _is_latin_base(char)
    return unicodedata.normalize("NFC", "".join(output))


def normalize_for_policy(text: str, policy: str) -> str:
    """Return the exact representation written to an FTS index/query."""

    if policy == "fold":
        return fold_latin_accents(text)
    if policy == "keep":
        return unicodedata.normalize("NFC", text)
    raise ValueError(f"unsupported accent policy: {policy!r}")


def query_terms(raw_query: str, policy: str) -> list[str]:
    """Tokenise a user query into safe, literal FTS words.

    FTS5 query syntax is never taken from the user.  Every extracted word is
    emitted as a quoted string and all words are combined with ``AND``.
    """

    normalised = normalize_for_policy(raw_query, policy)
    return _WORD_RE.findall(normalised)


def build_match_expression(raw_query: str, policy: str) -> tuple[str, list[str]]:
    """Build a literal, injection-safe FTS5 MATCH expression."""

    terms = query_terms(raw_query, policy)
    if not terms:
        raise ValueError("query must contain at least one searchable word")
    expression = " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
    return expression, terms
