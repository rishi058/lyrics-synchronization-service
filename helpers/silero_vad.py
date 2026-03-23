import torch 
import time
import gc 
from helpers.config import DEVICE, SAMPLE_RATE

# Minimum VAD segment duration (seconds) to be considered real speech.
# Filters out short instrument transients / false positives.
_MIN_SPEECH_DURATION = 0.3

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


def _detect_vocal_bounds(audio, audio_duration: float) -> tuple[float, float]:
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

