"""
process_hi.py
─────────────
Hindi/Hinglish lyrics processing pipeline.

Flow:
  1. Lyrics → transliterate + LLM refine → mixed_words [{lat, dev, lang}, ...]
  2. Filter Hindi words → align with Hindi model → hindi_aligned_words
  3. Use english_gap_filler to place English words in gaps between Hindi words
  4. Merge both lists sorted by timestamp
"""

import whisperx
import time
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_HI, SAMPLE_RATE
from helpers.hi.transliteration import is_devanagari
from helpers.utils import clean_for_alignment
from helpers.hi.process_helper import process_devanagari_script, process_latin_script
from helpers.silero_vad import detect_vocal_bounds

def process_hindi_language(lyrics: str, devanagari_output: bool, audio) -> list[dict]:
    """Returns sync data for Hindi/Hinglish language."""

    audio_duration = len(audio) / SAMPLE_RATE

    if lyrics:
        has_devanagari = is_devanagari(lyrics)

        if has_devanagari:
            lines = [clean_for_alignment(line, "devanagari") for line in lyrics.splitlines() if line.strip()]
            words_data, word_mapp = process_devanagari_script(lines)
        else:  # Latin script (Hinglish)
            lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
            words_data, word_mapp = process_latin_script(lines)
    else:
        segments = _transcribe_with_whisperx(audio)
        if not segments:
            raise RuntimeError("Hindi transcription failed: no segments returned.")
        full_text = " ".join(seg["text"] for seg in segments)
        lines = [clean_for_alignment(line, "devanagari") for line in full_text.splitlines() if line.strip()]
        words_data, word_mapp = process_devanagari_script(lines)

    # words_data = [{"lat":__, "dev":__, "lang":__}, ...]

    # ── Step 1: Build text for alignment (covers ALL words in Devanagari) ────
    align_text = _build_alignment_text(words_data)

    if not align_text.strip():
        raise RuntimeError("No words found in lyrics after processing.")

    # ── Step 2: Detect vocal bounds in a SINGLE VAD pass ────────────────────
    vocal_start, vocal_end = detect_vocal_bounds(audio, audio_duration)

    aligned_segments = [{"text": align_text, "start": vocal_start, "end": vocal_end}]

    # ── Step 3: Align ALL words in a single pass ─────────────────────────────
    try:
        print(f"[{time.strftime('%X')}] Aligning all words (Hindi + Transliterated English)...")
        model_a, metadata = whisperx.load_align_model(
            language_code="hi", device=DEVICE, model_name=ALIGN_MODEL_HI
        )
        result_aligned = whisperx.align(aligned_segments, model_a, metadata, audio, DEVICE)
        aligned_words = result_aligned["word_segments"]
    except Exception as e:
        raise RuntimeError(f"Alignment failed: {e}") from e

    if not aligned_words:
        raise RuntimeError("Alignment produced no word-level timestamps.")

    # ── Step 4: Output format ─────────────────────────────────────────────────
    if devanagari_output: 
        # output text should be in latin for "lang"="en" tag
        for item in aligned_words:
            word_key = item.get("word", item.get("text", ""))
            if word_key in word_mapp and word_mapp[word_key]["lang"] == "en":
                item["text"] = word_mapp[word_key]["lat"] 
            else:
                item["text"] = word_key

        return aligned_words
    else:
        # convert everything to latin
        for item in aligned_words:
            word_key = item.get("word", item.get("text", ""))
            if word_key in word_mapp:
                item["text"] = word_mapp[word_key]["lat"] 
            else:
                item["text"] = word_key
                
        return aligned_words


# ──────────────────────────────────────────────────────────────────────────────
# Private Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_alignment_text(mixed_words: list[dict]) -> str:
    """Extracts Devanagari representation for ALL words joined by spaces."""
    parts = [w["dev"] for w in mixed_words]
    return " ".join(parts).strip()

def _transcribe_with_whisperx(audio) -> list[dict]:
    """Transcribes audio using WhisperX in Hindi mode."""
    try:
        model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE)
        result = model.transcribe(audio, batch_size=16, chunk_size=10, language="hi")
        return result.get("segments", [])
    except Exception as e:
        raise RuntimeError(f"Hindi transcription failed: {e}") from e