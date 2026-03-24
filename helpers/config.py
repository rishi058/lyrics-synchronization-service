"""
config.py
─────────
Global constants and compiled regex patterns shared across all modules.
"""

import re
import torch

# ── Model size reference ──────────────────────────────────────────────────────
# | Model            | Disk   | Mem      |
# |------------------|--------|----------|
# | tiny             | 75 MB  | ~390 MB  |
# | base             | 142 MB | ~500 MB  |
# | small            | 466 MB | ~1.0 GB  |
# | medium           | 1.5 GB | ~2.6 GB  |
# | large-v3         | 2.9 GB | ~4.7 GB  |
# | large-v3-turbo   | 1.5 GB | ~4.7 GB  |

# ── Device & Model ────────────────────────────────────────────────────────────
DEVICE       = "cpu" # "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
COMPUTE_TYPE = "int8" # "float16" # use int8 if you want to save more memory (may reduce quality)
MODEL_NAME   = "large-v3-turbo" if DEVICE == "cuda" else "medium"

# ── Regex Patterns ────────────────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE      = re.compile(r'[a-zA-Z]')

# ── Audio ─────────────────────────────────────────────────────────────────────
SUPPORTED_AUDIO_EXTS = {".mp4", ".mp3", ".wav"}
SAMPLE_RATE          = 16000  # WhisperX uses 16kHz

# ── Alignment Models ──────────────────────────────────────────────────────────
ALIGN_MODEL_EN = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
ALIGN_MODEL_HI = "theainerd/Wav2Vec2-large-xlsr-hindi"

# ── Gap-Filling ───────────────────────────────────────────────────────────────
MIN_GAP_SECONDS = 0.05  # Ignore gaps smaller than 50ms (likely noise)
