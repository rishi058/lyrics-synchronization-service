from helpers.utils import clean_for_alignment
from llm.llm_service import LLMService

def process_en_language(segment_data: list[dict], lyrics: str, song_name: str) -> list[dict]:
    """
    segment_data: flat list of {"start", "end", "audio", "text"} dicts.
    If lyrics are provided, use the LLM to fit each segment's text to the lyrics.
    """
    if lyrics:
        transcribed_texts = [seg["text"] for seg in segment_data]

        cleaned_lyrics = " ".join(
            clean_for_alignment(line, "latin")
            for line in lyrics.splitlines() if line.strip()
        )

        best_fit_lyrics = LLMService().refine_lyrics_segment(transcribed_texts, cleaned_lyrics, "en", song_name)

        for i, segment in enumerate(segment_data):
            segment["text"] = best_fit_lyrics[i] if i < len(best_fit_lyrics) else segment["text"]

    return segment_data