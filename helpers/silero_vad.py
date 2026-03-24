import torch 
import time
import gc 
from helpers.config import DEVICE, SAMPLE_RATE

# Minimum VAD segment duration (seconds) to be considered real speech.
# Filters out short instrument transients / false positives.
_MIN_SPEECH_DURATION = 0.3
OVERLAP_SEC = 0.4   # overlap window to prevent word loss at hard cuts

def _load_silero_vad():
    """
    Load Silero VAD from torch.hub.

    Silero VAD is the underlying model whisperx uses internally.
    Calling it directly avoids depending on whisperx's private vad module.

    Returns:
        model                 callable VAD model
        get_speech_timestamps silero utility to extract timestamps
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=False,
    )
    # utils is a named tuple: (get_speech_timestamps, save_audio, read_audio,
    #                           VADIterator, collect_chunks)
    get_speech_timestamps = utils[0]
    return model, get_speech_timestamps

def detect_vocal_bounds(audio, audio_duration: float) -> tuple[float, float]:
    """
    Single-pass Silero VAD that returns (vocal_start, vocal_end).

    Uses silero-vad directly via torch.hub — no dependency on
    whisperx.vad (which is an internal, non-public module).

    Onset  : first VAD segment with duration >= _MIN_SPEECH_DURATION.
             Skips short transients / hi-hat hits / false positives.
    End    : last VAD segment end + 2-second safety buffer.

    Falls back to (0.0, audio_duration) on any error.
    """
    print(f"[{time.strftime('%X')}] Detecting vocal bounds (Silero VAD)...")

    vad_model = None
    try:
        vad_model, get_speech_timestamps = _load_silero_vad()
        vad_model = vad_model.to(DEVICE)

        # Silero expects a 1-D float32 tensor on the same device as the model.
        waveform = torch.tensor(audio, dtype=torch.float32).to(DEVICE)

        # return_seconds=True  → timestamps in seconds (not samples)
        # threshold            → confidence cutoff (0–1); 0.4 is a safe default
        # min_silence_duration_ms → merge segments separated by < this gap
        speech_segments: list[dict] = get_speech_timestamps(
            waveform,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            threshold=0.4,
            min_silence_duration_ms=300,
        )

    except Exception as e:
        print(f"[{time.strftime('%X')}] ⚠️  VAD failed: {e}. Falling back to full audio.")
        return 0.0, audio_duration

    finally:
        # Always free GPU memory regardless of success / failure.
        if vad_model is not None:
            del vad_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if not speech_segments:
        print(f"[{time.strftime('%X')}] ⚠️  VAD returned no segments. Using full audio.")
        return 0.0, audio_duration

    # ── Onset ────────────────────────────────────────────────────────────────
    # Walk segments in order; take the first one long enough to be real singing.
    vocal_start = 0.0
    for seg in speech_segments:
        duration = float(seg["end"]) - float(seg["start"])
        if duration >= _MIN_SPEECH_DURATION:
            vocal_start = max(0.0, float(seg["start"]))
            vocal_start = min(vocal_start, audio_duration * 0.5)  # cap at 50% of song
            break

    # ── End ──────────────────────────────────────────────────────────────────
    last_seg = speech_segments[-1]
    vocal_end = float(last_seg["end"]) + 2.0        # 2-second safety buffer
    vocal_end = min(vocal_end, audio_duration)       # never exceed file length
    vocal_end = max(vocal_end, vocal_start + 1.0)    # always strictly after start

    print(
        f"[{time.strftime('%X')}] Vocal bounds: "
        f"start={vocal_start:.2f}s  end={vocal_end:.2f}s"
    )
    return vocal_start, vocal_end

# START CHUNKING
def get_speech_chunks(audio) -> list[dict]:
    """
    Extracts raw speech segments from audio using Silero VAD.
    Returns:
        [{'start': float (seconds), 'end': float (seconds)}, ...]
    """
    print(f"[{time.strftime('%X')}] Extracting speech segments (Silero VAD)...")

    vad_model = None
    try:
        vad_model, get_speech_timestamps = _load_silero_vad()
        vad_model = vad_model.to(DEVICE)

        waveform = torch.tensor(audio, dtype=torch.float32).to(DEVICE)

        speech_segments: list[dict] = get_speech_timestamps(
            waveform,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            return_seconds=True,
            threshold=0.20,          # ← default is 0.5; lower catches faint/atypical vocals
            min_speech_duration_ms=80,   # ← catch short high notes
            min_silence_duration_ms=300, # ← don't over-split sustained notes
            speech_pad_ms=400,       # ← pad edges so note onset/release aren't clipped
        )

        """
        min_speech_duration_ms = any speech blob shorter than this is thrown away before segments are returned
        min_silence_duration_ms = If the silence between two blobs is shorter than this, Silero bridges them together as one continuous segment instead of splitting.
        speech_pad_ms = adds this much time to the start and end of each detected speech segment
        max_speech_duration_s = When a running speech region hits 12 seconds, Silero force-cuts it there and starts a new segment — regardless of whether silence was detected.
        """

    except Exception as e:
        print(f"[{time.strftime('%X')}] ⚠️  VAD failed to extract segments: {e}.")
        return []

    finally:
        if vad_model is not None:
            del vad_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    result = [{"start": float(seg["start"]), "end": float(seg["end"])} for seg in speech_segments]
    return _build_chunks(result)

# POST PROCESSING AFTER CHUNKING & TRANSCRIBING
def merge_overlapping_words(all_words: list[dict], overlap=OVERLAP_SEC) -> list[dict]:
    """
    After transcribing overlapping chunks, deduplicate words that appear twice
    near a cut boundary. Keep the one with higher confidence.
    """
    if not all_words:
        return []

    merged = [all_words[0]]
    for word in all_words[1:]:
        prev = merged[-1]
        # If timestamps overlap significantly — it's a duplicate
        if word["start"] < prev["end"] - (overlap / 2):
            # Keep higher confidence word
            if word.get("score", 0) > prev.get("score", 0):
                merged[-1] = word
        else:
            merged.append(word)
    return merged

#----------- PRIVATE HELPERS ----------------

# PASS 0
def _build_chunks(segments, max_gap=2.0, max_dur=25.0):
    # Pass 1 — non-negotiable splits at silence > 2s
    islands = _get_forced_boundaries(segments, max_gap=max_gap)
    
    # Pass 2 — duration splits within each island (with overlap)
    all_chunks = []
    for island in islands:
        all_chunks.extend(_chunk_island(island, max_dur=max_dur))
    
    return all_chunks

# PASS 1
def _get_forced_boundaries(segments, max_gap=2.0) -> list[list[dict]]:
    """
    Split segments into 'islands' wherever silence gap > 2s.
    These splits are non-negotiable — transcription accuracy depends on it.
    
    Returns: list of islands, each island = list of VAD segments
    """
    if not segments:
        return []

    islands = []
    current_island = [segments[0]]

    for seg in segments[1:]:
        gap = seg["start"] - current_island[-1]["end"]
        if gap > max_gap:
            islands.append(current_island)   # forced cut here
            current_island = [seg]
        else:
            current_island.append(seg)

    islands.append(current_island)
    return islands

# PASS 2
def _chunk_island(island: list[dict], max_dur=25.0, overlap=OVERLAP_SEC) -> list[dict]:
    """
    Within a single island (no gap > 2s inside):
    - Merge segments freely up to max_dur
    - If a hard cut is needed (single segment > max_dur), use overlap
    """
    chunks = []
    chunk_start = island[0]["start"]
    chunk_end   = island[0]["end"]

    for seg in island[1:]:
        projected = seg["end"] - chunk_start

        if projected <= max_dur:
            chunk_end = seg["end"]   # safe to merge
        else:
            # Hard cut needed — emit current chunk
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "hard_cut": False
            })
            # Start next chunk with overlap from previous end
            chunk_start = max(chunk_end - overlap, seg["start"])
            chunk_end   = seg["end"]

    # Flush
    chunks.append({
        "start": chunk_start,
        "end": chunk_end,
        "hard_cut": False
    })
    return chunks

