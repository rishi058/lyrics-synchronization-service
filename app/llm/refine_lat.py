from pydantic import BaseModel, Field
import time
from llm.base import BaseLLM

# ── Prompt ───────────────────────────────────────────────────────────────────
REFINE_LAT_PROMPT = """\
You are a Hindi/Urdu song-lyrics normaliser (Devanagari → Hinglish direction).

INPUT: list of {lat, dev, lang} dicts.
• dev = original Devanagari word (source of truth, may have minor transcription errors).
• lat = ITRANS transliteration (machine-generated, often unnatural).
• lang = auto-tagged 'hi' or 'en' (often wrong).

FIX EACH WORD:
1. lat  → rewrite to natural Hinglish spelling (e.g. 'aajaa'→'aja', 'mujhay'→'mujhe'). For English words use correct English spelling.
2. dev  → fix wrong matras/conjuncts/halant only. Cross-check against lat.
3. lang → 'hi' for Hindi/Urdu words, fillers (na, haan, re, ooh).
         → 'en' ONLY for real English words (love, baby, party, yeah, am, the).

After per-word fixes, verify sentence-level coherence — revert if a correction breaks meaning.

RULES:
- lang must be 'hi' or 'en' for every word, never empty.
- Preserve EXACT order & count — no merging, splitting, adding, or skipping.
- Never translate — only correct spelling/script.
- Every output object must have all 3 fields.\
"""


# ── Structured output models ────────────────────────────────────────────────
class RefinedWord(BaseModel):
    lat: str = Field(description="Natural Hinglish spelling")
    dev: str = Field(description="Corrected Devanagari spelling")
    lang: str = Field(description="'hi' or 'en'")


class RefinedLatWordsResponse(BaseModel):
    words: list[RefinedWord] = Field(description="Normalised word list (same order & count)")


# ── Mixin ────────────────────────────────────────────────────────────────────
class RefineLat(BaseLLM):
    def __init__(self):
        super().__init__()

    def refine_lat(self, lyrics: list[dict], song_name: str = "") -> list[dict]:
        if song_name:
            cached = self._load_llm_cache(f"{song_name}_llm_lat")
            if cached is not None:
                print(f"[{time.strftime('%X')}] [REFINE-DEV] Loaded from cache, Count: {len(cached)}")
                return cached
        result = self.invoke_chunked(
            items=lyrics,
            prompt=REFINE_LAT_PROMPT,
            response_format=RefinedLatWordsResponse,
            chunk_size=200,
            result_key="words",
            label="REFINE-LAT",
        )
        if song_name:
            self._save_llm_cache(f"{song_name}_llm_lat", result)
        return result