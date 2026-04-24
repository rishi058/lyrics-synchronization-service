"""
llm_service.py
──────────────
Single facade that exposes every LLM refinement method.
Instantiate once — the underlying ChatCohere client is a singleton.
"""

from llm.refine_dev import RefineDev
from llm.refine_lat import RefineLat
from llm.refine_lyrics_segment import RefineLyricsSegment


class LLMService(RefineDev, RefineLat, RefineLyricsSegment):
    """Unified entry-point for all LLM-powered refinement tasks."""

    _instance = None

    def __new__(cls):
        """Singleton — avoids re-creating the LLM client on every call."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only run parent init once
        if not hasattr(self, "_initialized"):
            super().__init__()
            self._initialized = True
