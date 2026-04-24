"""
models.py
─────────
Pydantic request/response models for the FastAPI endpoints.
"""

from typing import Literal
from pydantic import BaseModel


class SyncLyricsRequest(BaseModel):
    media_path: str
    output_path: str
    language: Literal["en", "hi"]
    lyrics: str = ""
    devanagari_output: bool = False    # Hindi only. False → ITRANS/Hinglish output