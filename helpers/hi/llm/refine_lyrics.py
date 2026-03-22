import time
from pydantic import BaseModel, Field
from helpers.logger import CustomLogger
from helpers.hi.llm.base import BaseLLM

REFINE_LYRICS_PROMPT = """\
You are a English-Hinglish-Hindi(Devanagari) language normaliser/refinement expert for Hindi/Urdu song lyrics.

CONTEXT:
You will receive a list of dictionary of words formatted as \
{{lat: 'original word:contains english word or hinglish word', dev: 'ITRANS transliteration of org in Devanagari', lang: 'hi or en determined using wordfreq py pkg'}} from Hinglish (Latin-script Hindi) song lyrics.

TASK:
Iterate through each word and:
1. Fix 'lang' tag('en' or 'hi'):
   - MOST words in these lyrics are HINDI/URDU words (lang='hi').
   - Words like 'Aja', 'Mujhay', 'apni', 'jhalak', 'dhikla', 'pahelio', 'ko' are ALL Hindi/Urdu words and MUST be tagged as lang='hi'.
   - if 'lat' contains a standard English dictionary word(Noun, Verb, Article, Adjective, Adverb, Pronoun, Preposition, Conjunction)(e.g., 'a', 'am', 'love', 'baby', 'party', 'yeah', 'crazy') Must be tagged as lang='en'.
2. Refine the 'dev' Devanagari-script to the CORRECT spelling.
   Fix any ITRANS conversion errors.
   Examples: 'Mujhay' → 'मुझे', 'pyaar' → 'प्यार', 'tera' → 'तेरा', 'Aja' → 'आजा'.
Then also Iterate by segments:
1. fix it if it doesn't look correct sentence-wise.
RULES:
- You MUST set 'lang' to either 'hi' or 'en' for EVERY word(MUST not be empty).
- Preserve the EXACT order of words from the input.
- Do NOT merge, split, or skip any words.
- Never translate only correct spelling, pronounciation should be preserved.
- Do NOT output empty objects. Every object in the list MUST contain all required(3) fields.
"""


# Structured Output Models
class RefinedWord(BaseModel):
    lat: str = Field(description="The exact word as it appeared in the input.")
    dev: str = Field(description="Devanagari script")
    lang: str = Field(description="'hi' for Hindi words, 'en' for English words.")


class RefinedLyrics(BaseModel):
    words: list[RefinedWord] = Field(description="List of normalized words preserving exact order.")


class RefineHinglishSong(BaseLLM):
    def __init__(self):
        super().__init__()

    def refine_hinglish(self, lyrics: list[dict]) -> list[dict]:
        print(f"[{time.strftime('%X')}] Total words before LLM: {len(lyrics)}")
        CustomLogger.log(f"--- [REFINING] LLM INPUT ---\n{lyrics}")

        chunk_size = 120
        chunks = [lyrics[i:i + chunk_size] for i in range(0, len(lyrics), chunk_size)]
        all_results: list[dict] = []

        for idx, chunk in enumerate(chunks):
            print(f"[{time.strftime('%X')}] Processing chunk {idx + 1}/{len(chunks)}...")
            try:
                result = self.invoke(REFINE_LYRICS_PROMPT, RefinedLyrics, chunk)
                all_results.extend(word.model_dump() for word in result.words)
                print(f"[{time.strftime('%X')}] Chunk {idx + 1} processed successfully.")
            except Exception as e:
                print(f"[{time.strftime('%X')}] ⚠️ LLM REFINING failed for chunk {idx + 1}: {e}")
                # Fallback: keep original words for this chunk instead of losing them
                print(f"[{time.strftime('%X')}] Using original (un-refined) words for chunk {idx + 1}.")
                all_results.extend(chunk)

        print(f"[{time.strftime('%X')}] Total words after LLM: {len(all_results)}")
        CustomLogger.log(f"--- [REFINING] LLM OUTPUT ---\n{all_results}")
        return all_results
