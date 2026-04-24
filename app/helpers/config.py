"""
config.py
─────────
Global constants and compiled regex patterns shared across all modules.
"""

import re
import os
import torch
from dotenv import load_dotenv
load_dotenv()

# ── Device & Model ────────────────────────────────────────────────────────────
DEVICE       = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"

# ── Regex Patterns ────────────────────────────────────────────────────────────
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
LATIN_RE      = re.compile(r'[a-zA-Z]')

# ── Audio ─────────────────────────────────────────────────────────────────────
SUPPORTED_AUDIO_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
SAMPLE_RATE          = 16000  # WhisperX uses 16kHz

# ── Tokens ─────────────────────────────────────────────────────────
COHERE_API_KEY  = os.getenv("COHERE_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "") 