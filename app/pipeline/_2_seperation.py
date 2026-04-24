import gc
import os
import logging
import torch

logger = logging.getLogger(__name__)


# ── Step 2: Remove backing vocals from an isolated vocal track ──────────────
def _remove_backing_vocals(raw_vocals_path: str, cache_dir: str,
                           song_name: str) -> str:
    """
    Takes a vocals-only WAV (44.1 kHz, from BS-RoFormer) and removes
    background vocals / harmonies / ad-libs using Mel-Band RoFormer Karaoke.

    Returns the path to a lead-vocals-only WAV at the model's native rate
    (44.1 kHz).  Caller is responsible for resampling to 16 kHz afterwards.

    Uses mel_band_roformer_karaoke_aufr33_viperx (SDR 10.19).
    This model uses the MDXC architecture (same as BS-RoFormer), which
    handles memory correctly — unlike the MDX architecture models (e.g.
    UVR_MDXNET_KARA_2.onnx) which OOM on files longer than ~30 seconds
    due to a bug in audio-separator 0.44.1's secondary stem inversion.
    """
    from audio_separator.separator import Separator

    lead_vocals_path = os.path.join(cache_dir, f"{song_name}_lead_raw.wav")

    #! Will download ~1 GB model on first run
    separator = Separator(
        output_dir=cache_dir,
        output_format="WAV",

        output_single_stem="Vocals",    # Keep only the lead vocal stem

        normalization_threshold=1.0,    # No normalization (same as Step 1)
        invert_using_spec=True,         # Spectrogram inversion for cleaner output
        use_autocast=True,              # Mixed-precision GPU inference

        # MDXC architecture params — same family as BS-RoFormer, no OOM bug
        mdxc_params={
            "segment_size": 256,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
            "override_model_segment_size": False,
        },
    )

    separator.load_model(
        model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
    )

    logger.info(f"Removing backing vocals with MelBand-RoFormer: {raw_vocals_path}")
    output_files = separator.separate(
        raw_vocals_path,
        {"Vocals": f"{song_name}_lead_raw"}
    )
    logger.info(f"BVoc removal complete: {output_files}")

    # Free VRAM / RAM
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not os.path.exists(lead_vocals_path):
        raise FileNotFoundError(
            f"BVoc removal failed — expected lead vocal at {lead_vocals_path}, "
            f"got: {output_files}"
        )

    return lead_vocals_path


def separate_vocals(input_path: str, remove_bvoc: bool = True) -> str:
    """
    Separates vocals from an audio file using BS-RoFormer via audio-separator.
    Returns path to the separated vocals WAV file (16 kHz, mono).

    BS-RoFormer (SDR 12.97) produces cleaner vocals than Demucs (SDR 10.8),
    preserving vocal timbre without the husky/robotic artifacts that degrade
    downstream ASR accuracy.

    Args:
        input_path:  Path to the ingested audio file (16 kHz mono WAV).
        remove_bvoc: If True (default), run a second pass with
                     Mel-Band RoFormer Karaoke to strip background vocals,
                     harmonies, and ad-libs from the isolated vocal track.
    """
    # 1. Path setup
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    cache_dir = os.path.join(root_dir, "cache", "seperations")
    os.makedirs(cache_dir, exist_ok=True)

    final_path = os.path.join(cache_dir, f"{song_name}_vocals.wav")

    # 2. Cache check
    if os.path.exists(final_path):
        return final_path

    #! Will Download 700 MB Model on first run
    # 3. Separate vocals using BS-RoFormer
    import subprocess
    from audio_separator.separator import Separator

    # ── Intermediate path (native 44.1kHz output from model) ─────────────────
    # BS-RoFormer was trained at 44.1kHz. Asking it to output 16kHz directly
    # causes internal resampling BEFORE separation — destroying high harmonics
    # (vibrato, overtones) that are critical for opera/classical singing.
    # We always let the model work at its native rate, then resample after.
    raw_vocals_path = os.path.join(cache_dir, f"{song_name}_vocals_raw.wav")

    separator = Separator(
        output_dir=cache_dir,
        output_format="WAV",

        # Do NOT set sample_rate=16000 here — that resamples BEFORE the model
        # sees the audio, destroying high-frequency vocal content.
        # We resample to 16kHz with ffmpeg AFTER separation instead.

        output_single_stem="Vocals",   # Skip instrumental stem output

        # Disable normalization — aggressive normalization (default 0.9) can
        # silence low-energy passages (opera pianissimo, falsetto) by clipping
        # the model's internal gain staging.
        normalization_threshold=1.0,

        # Spectrogram-domain inversion recovers harmonics that waveform
        # subtraction smears — essential for opera's dense overtone structure.
        invert_using_spec=True,

        use_autocast=True,             # Mixed-precision GPU inference (faster)

        # BS-RoFormer is an MDXC architecture model.
        # segment_size=384 → larger context window = fewer boundary artifacts
        # overlap=8        → maximum safe overlap for RoFormer (no skip/mute)
        mdxc_params={
            "segment_size": 384,
            "batch_size": 1,
            "overlap": 8,
            "pitch_shift": 0,
            "override_model_segment_size": True,
        },
    )

    # BS-RoFormer ep_317 SDR 12.97 — current gold standard for vocal isolation
    separator.load_model(
        model_filename="model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    )

    logger.info(f"Separating vocals with BS-RoFormer: {input_path}")
    output_files = separator.separate(
        input_path,
        {"Vocals": f"{song_name}_vocals_raw"}   # raw = 44.1kHz native output
    )
    logger.info(f"Separation complete (raw): {output_files}")

    # 4. Cleanup model from VRAM / RAM before next step
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not os.path.exists(raw_vocals_path):
        raise FileNotFoundError(
            f"Vocal separation failed — expected raw output at {raw_vocals_path}, "
            f"got: {output_files}"
        )

    # ── 4b. (Optional) Remove backing vocals / harmonies / ad-libs ───────────
    # Run Mel-Band RoFormer Karaoke on the isolated vocals to strip BVoc
    # layers, leaving only the lead singer.  This improves downstream ASR
    # accuracy by removing overlapping harmony words and ad-lib phantom
    # detections.  Uses MDXC architecture — no OOM issues.
    resample_input = raw_vocals_path          # default: skip BVoc removal
    lead_vocals_path = None

    if remove_bvoc:
        lead_vocals_path = _remove_backing_vocals(
            raw_vocals_path, cache_dir, song_name
        )
        resample_input = lead_vocals_path

    # 5. Resample 44.1kHz → 16kHz mono 16-bit PCM (AFTER model inference)
    #    This is where 16kHz enters — never before the model.
    logger.info(f"Resampling vocals to 16kHz mono: {final_path}")
    subprocess.run(
        [
            "ffmpeg", "-i", resample_input,
            "-ar", "16000",        # Downsample to 16 kHz (for ASR/VAD)
            "-ac", "1",            # Mono
            "-sample_fmt", "s16",  # 16-bit PCM
            "-y", final_path,
        ],
        check=True,
        capture_output=True,
    )

    # 6. Remove all intermediate 44.1kHz files now that final is written
    if os.path.exists(raw_vocals_path):
        os.remove(raw_vocals_path)
    if remove_bvoc and lead_vocals_path and os.path.exists(lead_vocals_path):
        os.remove(lead_vocals_path)

    logger.info(f"Separation complete: {final_path}")

    return final_path


# For checking as independent module
# if __name__ == "__main__":
#     print(separate_vocals(r"D:\STUDY 2\test\cache\ingestions\test.wav"))