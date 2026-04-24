import torch
import time
import numpy as np
import librosa
import logging
import os
import json
from helpers.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Downloads a small model in this path
# C:\Users\<username>/.cache\torch\hub\snakers4_silero-vad_master

# ── Silero VAD Loader ───────────────────────────────────────────────────────
def _load_vad():
    logger.info("Loading Silero VAD model …")
    model, utils = torch.hub.load(
        'snakers4/silero-vad', 'silero_vad', trust_repo=True
    )
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps


# ── VAD Parameters ──────────────────────────────────────────────────────────
# Aggressively tuned to catch ALL vocals across:
#   - Hindi-Urdu songs      (breathy, melismatic, soft-onset phrases)
#   - Afro-music / afrobeat (falsetto layers, off-beat phrasing, reverb tails)
#   - Rap / trap            (rapid-fire words with sub-100ms gaps between them)
#
# Strategy:
#   - Very low threshold   → model stays active for quiet/breathy moments
#   - Large min_silence    → short natural gaps (breath, beat rest) don't cut a segment
#   - Large pad            → onset/offset of every syllable is fully captured
#   - Short min_speech     → a single clipped syllable still becomes its own segment
#   - Wide merge gap       → musical breaks between rap bars / afro phrases are bridged

VAD_THRESHOLD          = 0.20   # very sensitive – catches falsetto, breathy Hindi ends, soft rap
VAD_MIN_SPEECH_MS      = 100    # keep even ultra-short syllable bursts (rap monosyllables)
VAD_MIN_SILENCE_MS     = 400    # require 400 ms of true silence to close a segment
VAD_SPEECH_PAD_MS      = 200    # pad 200 ms on both sides – wide enough for reverb tails

# Chunk boundary rules (seconds)
MAX_CHUNK_DURATION     = 20.0   # force-split longer vocal sections
MIN_CHUNK_DURATION     = 0.2    # keep very short trailing syllables / staccato rap words
MERGE_GAP_THRESHOLD    = 0.800  # bridge gaps ≤ 800 ms (rap bar rests, Afro call-response)

# Tail-guard: if the last segment ends within this many seconds of the file
# end, extend it all the way to the file end to capture fading/reverb vocals.
TAIL_GUARD_SEC         = 5.0    # generous – long reverb tails in Afro/Urdu studio records

# Head-guard: if the first segment starts after this many seconds from the
# file start, prepend from 0 so we don't miss intro vocals / pick-up phrases.
HEAD_GUARD_SEC         = 3.0    # if first segment starts after 3 s, pull it back to 0

# ── Public API ──────────────────────────────────────────────────────────────
def vad_chunking(media_path: str) -> list[dict]:
    """
    Loads audio from *media_path*, runs Silero VAD and returns a list of
    speech/vocal chunks with their timestamps and raw audio data.

    Returns:
        [
            {"start": 5.00, "end": 10.00, "audio": np.ndarray},
            {"start": 14.00, "end": 19.00, "audio": np.ndarray},
            ...
        ]

    Edge cases handled
    ------------------
    Completely silent file         → returns []
    Single continuous segment      → returns one chunk (split at MAX_CHUNK_DURATION)
    Soft singing / falsetto        → low VAD_THRESHOLD prevents false silence
    Vibrato / breath gaps          → MERGE_GAP_THRESHOLD bridges short silences
    Very short noise blips         → MIN_CHUNK_DURATION filter drops them
    SAMPLE_RATE mismatch           → librosa resamples on load
    Mono / stereo input            → converted to mono automatically
    Long unbroken vocal sections   → split into ≤MAX_CHUNK_DURATION sub-chunks
    """

    # ── Path setup and caching ──────────────────────────────────────────────
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(media_path))[0]
    cache_dir = os.path.join(root_dir, "cache", "vad_chunks")
    os.makedirs(cache_dir, exist_ok=True)
    cache_json = os.path.join(cache_dir, f"{song_name}_vad.json")
    cache_npz  = os.path.join(cache_dir, f"{song_name}_vad_audio.npz")

    if os.path.exists(cache_json) and os.path.exists(cache_npz):
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        audio_store = np.load(cache_npz)

        print(f"[{time.strftime('%X')}] Loading cached VAD chunks, Total Chunks: {len(meta)}") 

        return [
            {**seg, "audio": audio_store[str(i)]}
            for i, seg in enumerate(meta)
        ]

    # ── 1. Load & normalise audio ───────────────────────────────────────────
    logger.info(f"Loading audio: {media_path}")
    try:
        audio_np, sr = librosa.load(media_path, sr=SAMPLE_RATE, mono=True)
    except Exception as exc:
        logger.error(f"Failed to load audio file '{media_path}': {exc}")
        raise

    total_duration = len(audio_np) / SAMPLE_RATE
    logger.info(f"Audio duration: {total_duration:.2f}s  |  sample rate: {SAMPLE_RATE} Hz")

    if len(audio_np) == 0:
        logger.warning("Audio file is empty – returning no chunks.")
        
        # Cleanup
        audio_np = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return []

    # Silero VAD expects a 1-D float32 torch tensor.
    # torch.from_numpy already shares memory when dtypes match,
    # so only call .float() if the source isn't already float32.
    audio_tensor = torch.from_numpy(audio_np)
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()

    # ── 2. Run VAD ──────────────────────────────────────────────────────────
    model, get_speech_timestamps = _load_vad()

    logger.info("Running VAD …")
    try:
        raw_segments = get_speech_timestamps(
            audio_tensor,
            model,
            threshold           = VAD_THRESHOLD,
            sampling_rate       = SAMPLE_RATE,
            min_speech_duration_ms  = VAD_MIN_SPEECH_MS,
            min_silence_duration_ms = VAD_MIN_SILENCE_MS,
            speech_pad_ms       = VAD_SPEECH_PAD_MS,
            return_seconds      = False,   # get sample indices for precision
        )
    except Exception as exc:
        logger.error(f"Silero VAD failed: {exc}")
        raise

    if not raw_segments:
        logger.warning("VAD found no speech segments – audio may be completely silent.")
        
        # Cleanup RAM / VRAM
        model = None
        get_speech_timestamps = None
        audio_tensor = None
        audio_np = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return []

    logger.info(f"VAD raw segments: {len(raw_segments)}")

    # ── 3. Convert sample indices → seconds ─────────────────────────────────
    def _samples_to_sec(samples: int) -> float:
        return round(samples / SAMPLE_RATE, 4)

    segments = [
        {
            "start": _samples_to_sec(seg["start"]),
            "end":   _samples_to_sec(seg["end"]),
        }
        for seg in raw_segments
    ]

    # ── 4. Merge segments whose gap is below MERGE_GAP_THRESHOLD ────────────
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        gap = seg["start"] - merged[-1]["end"]
        if gap <= MERGE_GAP_THRESHOLD:
            # Capture values *before* mutation for accurate logging
            prev_start = merged[-1]["start"]
            prev_end   = merged[-1]["end"]
            # bridge the gap (vibrato / breath between phrases)
            merged[-1]["end"] = seg["end"]
            logger.debug(
                f"  Merged gap {gap:.3f}s between "
                f"{prev_start:.2f}s–{prev_end:.2f}s and "
                f"{seg['start']:.2f}s–{seg['end']:.2f}s"
            )
        else:
            merged.append(seg.copy())

    logger.info(f"After merging short gaps: {len(merged)} segment(s)")

    # ── 4b. Head-guard: pull first segment back to 0 if vocals start early ──
    # Pick-up phrases, a cappella intros, or Afro call-in vocals that begin
    # before the first detected segment are captured by anchoring to 0.
    if merged:
        head_gap = merged[0]["start"]
        if 0 < head_gap <= HEAD_GUARD_SEC:
            old_start = merged[0]["start"]
            merged[0]["start"] = 0.0
            logger.info(
                f"Head-guard: pulled first segment start from "
                f"{old_start:.2f}s → 0.00s "
                f"(gap was {head_gap:.2f}s)"
            )

    # ── 4c. Tail-guard: extend last segment to file end if close ────────────
    # Fading / reverb tails at the end of a song often fall below VAD_THRESHOLD
    # and are simply not returned by VAD at all.  If the last detected segment
    # ends within TAIL_GUARD_SEC of the actual file end, stretch it so we
    # capture every remaining sample.
    if merged:
        tail_gap = total_duration - merged[-1]["end"]
        if 0 < tail_gap <= TAIL_GUARD_SEC:
            old_end = merged[-1]["end"]
            merged[-1]["end"] = round(total_duration, 4)
            logger.info(
                f"Tail-guard: extended last segment end from "
                f"{old_end:.2f}s → {merged[-1]['end']:.2f}s "
                f"(gap was {tail_gap:.2f}s)"
            )

    # ── 5. Split segments exceeding MAX_CHUNK_DURATION ──────────────────────
    # Instead of splitting at a hard boundary, look for a local energy minimum
    # in a search window around the ideal split point to avoid cutting
    # mid-phoneme / mid-word / mid-note.
    SPLIT_SEARCH_SEC = 1.0   # look ±1 s around the ideal boundary
    ENERGY_FRAME_SEC = 0.01  # 10 ms RMS frames for the energy curve

    def _find_energy_min_split(seg_start: float, ideal_split: float,
                               seg_end: float) -> float:
        """Return a split point near *ideal_split* that sits at a local
        energy minimum, so we cut during a quiet moment rather than
        mid-phoneme."""
        search_lo = max(seg_start, ideal_split - SPLIT_SEARCH_SEC)
        search_hi = min(seg_end,   ideal_split + SPLIT_SEARCH_SEC)

        lo_sample = int(search_lo * SAMPLE_RATE)
        hi_sample = int(search_hi * SAMPLE_RATE)
        lo_sample = max(0, min(lo_sample, len(audio_np) - 1))
        hi_sample = max(lo_sample + 1, min(hi_sample, len(audio_np)))

        window = audio_np[lo_sample:hi_sample]
        if len(window) == 0:
            return ideal_split

        frame_len = max(1, int(ENERGY_FRAME_SEC * SAMPLE_RATE))
        n_frames  = len(window) // frame_len
        if n_frames == 0:
            return ideal_split

        # Compute per-frame RMS energy
        trimmed  = window[: n_frames * frame_len].reshape(n_frames, frame_len)
        rms      = np.sqrt(np.mean(trimmed ** 2, axis=1))

        min_idx   = int(np.argmin(rms))
        best_sample = lo_sample + min_idx * frame_len + frame_len // 2
        return best_sample / SAMPLE_RATE

    def _split_long(seg: dict) -> list[dict]:
        """Split a single segment into ≤MAX_CHUNK_DURATION sub-segments,
        preferring local energy minima as split points."""
        duration = seg["end"] - seg["start"]
        if duration <= MAX_CHUNK_DURATION:
            return [seg]
        pieces = []
        cursor = seg["start"]
        while cursor < seg["end"]:
            remaining = seg["end"] - cursor
            if remaining <= MAX_CHUNK_DURATION:
                pieces.append({"start": cursor, "end": seg["end"]})
                break
            # Ideal boundary at MAX_CHUNK_DURATION; refine via energy
            ideal = cursor + MAX_CHUNK_DURATION
            split_at = _find_energy_min_split(cursor, ideal, seg["end"])
            # Safety: ensure forward progress of at least 1 s
            split_at = max(split_at, cursor + 1.0)
            split_at = min(split_at, seg["end"])
            pieces.append({"start": cursor, "end": split_at})
            cursor = split_at
        return pieces

    split_segments: list[dict] = []
    for seg in merged:
        split_segments.extend(_split_long(seg))

    # ── 6. Filter out very short blips (< MIN_CHUNK_DURATION) ───────────────
    final_segments = [
        seg for seg in split_segments
        if (seg["end"] - seg["start"]) >= MIN_CHUNK_DURATION
    ]
    dropped = len(split_segments) - len(final_segments)
    if dropped:
        logger.info(f"Dropped {dropped} micro-segment(s) shorter than {MIN_CHUNK_DURATION}s")

    if not final_segments:
        logger.warning("No valid chunks remain after filtering.")
        
        # Cleanup RAM / VRAM
        model = None
        get_speech_timestamps = None
        audio_tensor = None
        audio_np = None
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return []

    # ── 7. Slice audio and build output ─────────────────────────────────────
    chunks: list[dict] = []
    for i, seg in enumerate(final_segments):
        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample   = int(seg["end"]   * SAMPLE_RATE)

        # Guard against float-rounding going out of bounds
        start_sample = max(0, min(start_sample, len(audio_np) - 1))
        end_sample   = max(start_sample + 1, min(end_sample, len(audio_np)))

        chunk_audio = audio_np[start_sample:end_sample].copy()

        # ── Derive end from actual slice length, NOT from the VAD/tail-guard ──
        # The tail-guard (and float arithmetic in general) can make seg["end"]
        # differ slightly from the true audio boundary.  ForcedAligner uses
        # (end - start) as the expected duration of the audio it receives, so
        # any mismatch causes alignment drift.  Computing end from sample count
        # guarantees:  end - start  ==  len(audio) / SAMPLE_RATE  exactly.
        actual_end = round(seg["start"] + len(chunk_audio) / SAMPLE_RATE, 6)

        chunks.append({
            "start": seg["start"],
            "end":   actual_end,
            "audio": chunk_audio,          # np.ndarray, float32, mono, SAMPLE_RATE Hz
        })

        
    print(f"[{time.strftime('%X')}] Total Chunks Created: {len(chunks)}")
    
    # ── 8. Cleanup RAM / VRAM ───────────────────────────────────────────────
    # Remove references to large local structures to allow GC to clear them
    model = None
    get_speech_timestamps = None
    audio_tensor = None
    audio_np = None

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save metadata (human-editable) and audio arrays separately
    meta = [{"start": c["start"], "end": c["end"]} for c in chunks]
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    np.savez(cache_npz, **{str(i): c["audio"] for i, c in enumerate(chunks)})
    logger.info(f"VAD cache saved → {cache_json} + {cache_npz}")

    return chunks
