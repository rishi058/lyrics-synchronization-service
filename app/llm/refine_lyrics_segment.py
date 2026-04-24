"""
refine_lyrics_segment.py
────────────────────────
Corrects WhisperX-generated segment text against user-provided lyrics.
Fills missing words, replaces misheard words, preserves segment boundaries.
"""

import time
from pydantic import BaseModel, Field
from llm.base import BaseLLM
from helpers.logger import CustomLogger

# ── Prompt ───────────────────────────────────────────────────────────────────
REFINE_LYRICS_SEGMENTS_PROMPT = """\
You are a lyrical-segment corrector for English and Hindi songs.
YOU ARE NOT A TRANSLATOR, NEVER TRANSLATE ANY WORDS(EVEN ABIGIOUS ONE)

INPUT:
• segmented_lyrics — ordered list of text segments produced by a ASR \
model. These may contain misheard words, dropped words, or informal \
short-forms. Each segment maps to a fixed audio time-range.
• lyrics — the complete, correct reference lyrics for this portion of the song \
(provided as a single string, NOT segmented).

ALWAYS ACHEIVE FINAL OUTPUT IN 2 STEPS:
1st Step: Correct every segment so its text matches the reference lyrics as closely as \
possible, while strictly respecting the segment boundaries described below.

2nd Step: Iterate throught the corrected segment and expand only eligible contraction and short-form to its full \

═══════════════════════════════════════════════════════════════
                STEP 1: REFINE SEGMENTS 
═══════════════════════════════════════════════════════════════
=> SEGMENT BOUNDARY RULES  (HIGHEST PRIORITY)

Each segment corresponds to a slice of audio. The words inside a segment \
must stay inside that segment — do NOT shift words across segment borders.

► NEVER INSERT new words BEFORE the first word of a segment.
► NEVER INSERT new words AFTER the last word of a segment.
► You MAY INSERT new words in b/w sentence of segment if its there in lyrics[start:end] and not in seg[start:end].
► You MAY REPLACE the first or last word if the transcriber misheard it (i.e. a word that sounds similar exists in \
the reference lyrics at that position). Replacement is NOT insertion.

EXAMPLE — CORRECT vs WRONG:
  original segment : "Well, it's a part. It ain't working"
  reference lyrics : "…pull us apart. It ain't working…"
  ✔ CORRECT output : "Pull us apart. It ain't working"
  (Well, it's a part) == (pull us apart): [misheard by transcriber]

  original segment : "Cause I love you."
  reference lyrics : "…yeah cause I love you yeah…"
  ✔ CORRECT output : "Cause I love you."
  (Yeah, cause I love you) == (Cause I love you): [Start & End word should be same]

► STICTLY FOLLOW: If reference-lyrics word-groups don't map to any existing part in segments, leave them out.

► CORRECTION RULES

1. FILL missing words BETWEEN existing words using the reference lyrics \
   (the transcriber often drops words in the middle of a segment).
2. REPLACE misheard words — words that sound similar but are transcribed \
   incorrectly due to accent or model error. This applies to ANY position \
   in the segment (start, middle, or end).
3. Preserve the EXACT number and order of segments. \
   Do NOT merge, split, add, or remove segments.
4. If a segment already matches the reference lyrics, return it UNCHANGED.

══════════════════════════════════════════════════════════════════
        STEP 2:  CONTRACTION / SHORT-FORM EXPANSION  (MANDATORY)
══════════════════════════════════════════════════════════════════
You MUST expand all eligible contraction and short-form to its full \
written form. This is required for the forced-alignment model to work.

  ✔ ALWAYS EXPAND (pronunciation is compatible):
     I'm → I am  ·  I've → I have  ·  I'll → I will  ·  I'd → I would
     he's → he is  ·  she's → she is  ·  it's → it is  ·  that's → that is

  ✘ DO NOT EXPAND (spoken sound changes too much):
     can't · won't · don't · didn't · couldn't · wouldn't · shouldn't
     isn't · aren't · hasn't · haven't · ain't · wasn't · weren't\
"""


REFINE_LYRICS_SEGMENTS_HI_PROMPT = """\
You are a lyrical-segment corrector for Hinglish (Hindi-Urdu mixed with English) songs.
YOU ARE NOT A TRANSLATOR, NEVER TRANSLATE ANY WORDS.

INPUT:
• segmented_lyrics — ordered list of text segments produced by an ASR \
model. These may contain misheard words, dropped words, or garbled \
code-switching artifacts. Each segment maps to a fixed audio time-range.
• lyrics — the complete, correct reference lyrics for this portion of the song \
(provided as a single string in Latin/Hinglish script, NOT segmented).

YOUR TASK:
Correct every segment so its text matches the reference lyrics as closely as \
possible, while strictly respecting the segment boundary rules below.

═══════════════════════════════════════════════════════════════
          SEGMENT BOUNDARY RULES  (HIGHEST PRIORITY)
═══════════════════════════════════════════════════════════════
Each segment corresponds to a slice of audio. The words inside a segment \
must stay inside that segment — do NOT shift words across segment borders.

► NEVER INSERT new words BEFORE the first word of a segment.
► NEVER INSERT new words AFTER the last word of a segment.
► You MAY INSERT new words in b/w sentence of segment if its there in lyrics[start:end] and not in seg[start:end].
► You MAY REPLACE the first or last word if the transcriber misheard it (i.e. a word that sounds similar exists in \
the reference lyrics at that position). Replacement is NOT insertion.

═══════════════════════════════════════════════════════════════
             CODE-SWITCHING RECOVERY  (CRITICAL)
═══════════════════════════════════════════════════════════════
Hinglish songs frequently switch between Hindi/Urdu and English mid-segment. \
The ASR model (set to Hindi mode) OFTEN GARBLES English phrases into \
Hindi-sounding phonemes. You MUST detect and recover these.

► GARBLED ENGLISH DETECTION:
  When you see a cluster of Hindi/Hinglish words at ANY position in a segment \
  (start, middle, or end) that:
    1. Does NOT make grammatical sense in Hindi/Urdu, AND
    2. Phonetically resembles an English phrase from the reference lyrics
  → REPLACE the garbled cluster with the correct English phrase from lyrics.

► COMMON PATTERN:
  ASR with language="Hindi" is MOST LIKELY to garble English phrases that appear at the beginning of a segment 
  EXAMPLE:   
    ASR segment:   "Mein nATa nahi hU.N Mein nATa nahi hU.N ..."
    reference:     "I'm not enough I'm not enough..."
    ✔ CORRECT:     "I am not enough I am not enough ..."    
    
    WHY: "Mein nATa nahi hU.N" ≈ phonetic Hindi rendering of "I'm not enough".
         The lyrics confirm this. Replace the garbled Hindi with the English original.
         "patA Hai Tu na vilI" maps to the next part — keep it (will be corrected by normal rules).

► PHONETIC MATCHING GUIDE for Hindi↔English garbling:
  - "Mein" / "mai" can be garbled from "I'm" or "my"
  - "nATa" / "naat" can be garbled from "not"
  - "nahi" can be garbled from "enough" (partial phonetic overlap)
  - Any nonsensical Hindi cluster near a segment boundary likely maps to an English phrase in the reference lyrics.

► PRESERVE ENGLISH AS-IS: If the ASR already correctly transcribed English words keep them exactly as-is.

► STICTLY FOLLOW: If reference-lyrics word-groups don't map to any existing part in segments, leave them out.
═══════════════════════════════════════════════════════════════
                    CORRECTION RULES
═══════════════════════════════════════════════════════════════
1. FILL missing words BETWEEN existing words using the reference lyrics \
   (the transcriber often drops words in the middle of a segment).
2. REPLACE misheard words — words that sound similar but are transcribed \
   incorrectly due to accent or model error. This applies to ANY position \
   in the segment (start, middle, or end).
3. Preserve the EXACT number and order of segments. \
   Do NOT merge, split, add, or remove segments.
4. If a segment already matches the reference lyrics, return it UNCHANGED.
"""


# ── Structured output ───────────────────────────────────────────────────────
class RefineLyricsSegmentResponse(BaseModel):
    refined_lyrics: list[str] = Field(description="Corrected segment texts (same count & order)")


# ── Mixin ────────────────────────────────────────────────────────────────────
class RefineLyricsSegment(BaseLLM):
    def __init__(self):
        super().__init__()

    def refine_lyrics_segment(
        self,
        segmented_lyrics: list[str],
        lyrics: str,
        language: str,
        song_name: str = "",
    ) -> list[str]:

        # ── Cache check ─────────────────────────────────────────────────────────
        if song_name:
            cache_name = f"{song_name}_{language}_llm_segments"
            cached = self._load_llm_cache(cache_name)
            if cached is not None:
                print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] Loaded from cache ({cache_name})")
                return cached
        
        seg_chunks = _chunk_segments(segmented_lyrics)
        lyr_chunks = _align_lyrics_to_chunks(seg_chunks, lyrics)
        all_results: list[str] = []

        ts = time.strftime('%X')
        print(f"[{ts}] [REFINE-SEGMENT] Total Chunks: {len(seg_chunks)}")

        for idx, chunk in enumerate(seg_chunks, 1):
            ts = time.strftime('%X') 

            # Mention exact length so the LLM tries harder to match it
            user_input = {
                "EXPECTED_OUTPUT_SEGMENTS_COUNT": len(chunk),
                "segmented_lyrics": chunk, 
                "lyrics": lyr_chunks[idx - 1]
            }

            prompt = REFINE_LYRICS_SEGMENTS_HI_PROMPT if language == "hi" else REFINE_LYRICS_SEGMENTS_PROMPT

            try:
                result = self.invoke(
                    prompt,
                    RefineLyricsSegmentResponse,
                    user_input,
                )
                all_results.extend(result.refined_lyrics)
                print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] chunk {idx} ✔, input-seg: {len(chunk)}, output-seg: {len(result.refined_lyrics)}")
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️ [REFINE-SEGMENT] chunk {idx} failed: {e}")
                print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] falling back to originals")
                all_results.extend(chunk)

        print(f"[{time.strftime('%X')}] [REFINE-SEGMENT] Completed — final segment count: {len(all_results)}")

        log_result = [] 
        for i, seg in enumerate(all_results):
            a = f"[{i+1}] {segmented_lyrics[i]}\n"
            b = f"[{i+1}] {all_results[i]}\n"
            log_result.append(a)
            log_result.append(b)
            log_result.append("\n")
                
        CustomLogger.log(f"--- [REFINE-SEGMENT] LLM INPUT VS OUTPUT ---\n{''.join(log_result)}")

        if song_name:
            self._save_llm_cache(cache_name, all_results)

        return all_results


# ── Helper: smart chunking ──────────────────────────────────────────────────

def _chunk_segments(
    sentences: list[str],
    target_words: int = 120,
    tolerance: int = 30,
) -> list[list[str]]:
    """Group sentences into chunks of roughly *target_words* (±tolerance)."""
    chunks: list[list[str]] = []
    current: list[str] = []
    word_count = 0

    for sentence in sentences:
        n = len(sentence.split())
        if current and word_count >= target_words and word_count + n > target_words + tolerance:
            chunks.append(current)
            current = []
            word_count = 0
        current.append(sentence)
        word_count += n

    if current:
        chunks.append(current)
    return chunks


def _align_lyrics_to_chunks(
    seg_chunks: list[list[str]],
    lyrics: str,
    buffer: int = 7,
) -> list[str]:
    """Slice *lyrics* proportionally to match each segment chunk (+ small buffer)."""
    words = lyrics.split()
    chunk_sizes = [sum(len(s.split()) for s in c) for c in seg_chunks]
    result: list[str] = []
    pos = 0

    for size in chunk_sizes:
        end = min(pos + size + buffer, len(words))
        result.append(" ".join(words[pos:end]))
        pos = end

    # Attach leftover words to the last chunk
    if pos < len(words) and result:
        result[-1] += " " + " ".join(words[pos:])

    return result