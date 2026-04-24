
from pydantic import BaseModel, Field
import time
from llm.base import BaseLLM

# ── Prompt ───────────────────────────────────────────────────────────────────
REFINE_DEV_PROMPT = """\
You are a Hindi/Urdu song-lyrics normaliser (Hinglish → Devanagari direction).

INPUT: list of {lat, dev, lang} dicts.
• lat = original Hinglish word (source of truth, keep as-is).
• dev = machine-transliterated Devanagari (may be wrong).
• lang = auto-tagged 'hi' or 'en' (often wrong).

FIX EACH WORD:
1. lang → 'hi' for any Hindi/Urdu word (aja, mujhe, dil, ko, na, haan, re, ooh).
         → 'en' ONLY for real English words (love, baby, party, yeah, am, the).
2. dev  → correct Devanagari spelling/matras (e.g. 'Mujhay'→'मुझे', 'Aja'→'आजा', 'pyaar'→'प्यार').
3. lat  → minimal or no changes (it's the original source).

After per-word fixes, verify sentence-level coherence.

RULES:
- lang must be 'hi' or 'en' for every word, never empty.
- Preserve EXACT order & count — no merging, splitting, adding, or skipping.
- Never translate — only correct script/spelling.
- Every output object must have all 3 fields.\
"""


# ── Structured output models ────────────────────────────────────────────────
class RefinedWord(BaseModel):
    lat: str = Field(description="Original Hinglish word (minimal changes)")
    dev: str = Field(description="Corrected Devanagari spelling")
    lang: str = Field(description="'hi' or 'en'")


class RefinedDevWordsResponse(BaseModel):
    words: list[RefinedWord] = Field(description="Normalised word list (same order & count)")


# ── Mixin ────────────────────────────────────────────────────────────────────
class RefineDev(BaseLLM):
    def __init__(self):
        super().__init__()

    def refine_dev(self, lyrics: list[dict], song_name: str = "") -> list[dict]:
        if song_name:
            cached = self._load_llm_cache(f"{song_name}_llm_dev")
            if cached is not None:
                print(f"[{time.strftime('%X')}] [REFINE-DEV] Loaded from cache, Count: {len(cached)}")
                return cached
        result = self.invoke_chunked(
            items=lyrics,
            prompt=REFINE_DEV_PROMPT,
            response_format=RefinedDevWordsResponse,
            chunk_size=200,
            result_key="words",
            label="REFINE-DEV",
        )
        if song_name:
            self._save_llm_cache(f"{song_name}_llm_dev", result)
        return result
