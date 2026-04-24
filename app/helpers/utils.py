import re
import unicodedata
from typing import Literal
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from helpers.config import DEVANAGARI_RE

#---------------------------------------------------------------------------------------------------

# {'key': {'lat':___, 'lang':___}, }
global_word_mapp = {}

#---------------------------------------------------------------------------------------------------

# Invisible/zero-width Unicode characters that look like spaces but aren't real word separators.
# These are inserted by lyrics websites between codepoints for line-breaking, causing .split()
# to treat each Devanagari matra/consonant as a separate "word".
_INVISIBLE_CHARS_RE = re.compile(
    r'['
    r'\u200B'   # Zero Width Space
    r'\u200C'   # Zero Width Non-Joiner (ZWNJ)
    r'\u200D'   # Zero Width Joiner (ZWJ)
    r'\u2060'   # Word Joiner
    r'\uFEFF'   # Zero Width No-Break Space (BOM)
    r'\u00AD'   # Soft Hyphen
    r']'
)

def _normalize_spaces(text: str) -> str:
    """Strip invisible/zero-width Unicode characters and normalise to NFC."""
    text = _INVISIBLE_CHARS_RE.sub('', text)
    return unicodedata.normalize('NFC', text)


def clean_for_alignment(text: str, script: Literal["latin", "devanagari"]) -> str:
    """
    Strips characters that alignment models cannot handle:
    1. keeps only ASCII letters, whitespace, and internal apostrophes for latin
    2. keeps only Devanagari codepoints and whitespace for devanagari
    Collapses multiple spaces and trims the result.

    Also strips invisible Unicode space-like characters (U+200B, ZWNJ, ZWJ, etc.)
    that lyrics websites embed between codepoints.  These are treated as word
    boundaries by Python's .split(), causing 'बिन' to split into ['ब', 'ि', 'न'].
    """
    text = _normalize_spaces(text)
    if script == "latin":
        cleaned = re.sub(r"[^a-zA-Z\s']", '', text)
        cleaned = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '', cleaned)
    else:
        # Keep only Devanagari codepoints (U+0900-U+097F) and plain ASCII whitespace.
        # Do NOT use \s here — it matches U+200B and other invisible separators that
        # would slip through and cause per-codepoint splitting downstream.
        cleaned = re.sub(r'[^\u0900-\u097F \t\n\r]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

#---------------------------------------------------------------------------------------------------
"""
[
    {"start": 5.00, "end": 10.00, "aligned_words" : [
        {"word": ".......", "start": 0.00, "end": 5.00},
        {"word": ".......", "start": 5.00, "end": 10.00}, ...
    ]},
    ...
]
"""
def _dev_to_itrans_fallback(token: str) -> str:
    """
    Last-resort fallback for a token not found in global_word_mapp.
    - Devanagari token  → transliterate to ITRANS (phonetic Latin)
    - Latin token       → return as-is (already Latin)
    """
    if DEVANAGARI_RE.search(token):
        return transliterate(token, sanscript.DEVANAGARI, sanscript.ITRANS)
    return token  # already Latin, nothing to do


def _lookup_word(raw: str, lang_filter: str | None = None) -> str | None:
    """
    Look up *raw* in global_word_mapp.
    If not found directly (e.g. words were merged with a space by timestamp merging),
    split by space, look up each token individually and rejoin.
    For tokens still not found, fall back to Devanagari→ITRANS transliteration so
    the result is always Latin — no raw Devanagari leaks through.
    Returns the mapped string, or None if lang_filter blocked every hit (single-token path).
    """
    def _single(token: str) -> str | None:
        entry = global_word_mapp.get(token)
        if entry is None:
            return None
        if lang_filter and entry["lang"] != lang_filter:
            return None
        return entry["lat"]

    # Direct hit
    result = _single(raw)
    if result is not None:
        return result

    # Fallback: try splitting on space (merged words, e.g. "I एम" after timestamp merging)
    tokens = raw.split(" ")
    if len(tokens) <= 1:
        # Single token, not in map.
        # If lang_filter is active we return None (caller decides whether to skip).
        # Without a filter, use direct transliteration so we never emit raw Devanagari.
        if lang_filter:
            return None
        return _dev_to_itrans_fallback(raw)

    mapped_tokens = []
    for token in tokens:
        mapped = _single(token)
        if mapped is not None:
            mapped_tokens.append(mapped)
        else:
            # Not in map — transliterate Devanagari → ITRANS, keep Latin as-is.
            mapped_tokens.append(_dev_to_itrans_fallback(token))

    return " ".join(mapped_tokens)


def format_segment_for_hindi(segment_data, devanagari_output) -> list[dict]:
    # currently segment_data has only dev script

    if devanagari_output:
        # only replace dev words which are actually english
        for seg in segment_data:
            for word in seg.get("aligned_words", []):
                mapped = _lookup_word(word["word"], lang_filter="en")
                if mapped is not None:
                    word["word"] = mapped

    else:
        # replace all dev words with their Latin equivalents.
        # _lookup_word guarantees Latin output via ITRANS fallback — no Devanagari leaks.
        for seg in segment_data:
            for word in seg.get("aligned_words", []):
                word["word"] = _lookup_word(word["word"])

    return segment_data