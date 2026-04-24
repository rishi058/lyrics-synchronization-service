import subprocess
import os
import shutil
import soundfile as sf

def ingest(input_path: str) -> str:
    """
    returns output_path
    """
    _check_ffmpeg()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(input_path))[0]
    
    os.makedirs(os.path.join(root_dir, "cache", "ingestions"), exist_ok=True)
    output_path = os.path.join(root_dir, "cache", "ingestions", f"{song_name}.wav")

    # 2. Caching System
    # If the file exists, read and return immediately
    if os.path.exists(output_path):
        return output_path

    # 3. Processing (FFmpeg)
    # Output: 16kHz mono 16-bit PCM WAV — audio-only, metadata preserved.
    # Size reduction vs original comes from: lower sample-rate + mono + 16-bit depth.
    cmd = [
        "ffmpeg", "-i", input_path,
        "-vn",                # Strip all video streams
        "-map_metadata", "0", # Preserve source file metadata (title, artist, album, etc.)
        "-ar", "16000",       # Downsample to 16 kHz (required by most ASR/VAD models)
        "-ac", "1",           # Downmix to mono
        "-sample_fmt", "s16", # 16-bit signed PCM (2 bytes/sample — smallest lossless WAV)
        "-y", output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 4. Post-processing & Validation
    data, sr = sf.read(output_path)
    duration = len(data) / sr

    assert 5 < duration < 600, f"File duration {duration:.1f}s is outside allowed limits."

    return output_path

#!-------------------------PRIVATE HELPERS------------------------------

def _check_ffmpeg():
    """Raise a helpful error if ffmpeg is not found on PATH."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError(
            "FFmpeg not found. Install it and ensure it's on your PATH.\n"
        )