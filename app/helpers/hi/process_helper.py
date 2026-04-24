from helpers.hi.transliteration import hinglish_to_devanagari, devanagari_to_hinglish
from llm.llm_service import LLMService
import helpers.utils as utility

def process_devanagari_script(lines: list[str], song_name: str) -> list[dict]:
    """
    Takes Devanagari lyrics lines, converts to Hinglish, refines via LLM and return it.
    """
    lyrics_text = " ".join(lines) 
    formatted_words = devanagari_to_hinglish(lyrics_text) # returns [{'lat':__, 'dev':__, 'lang':__}, ...]

    refined_words = LLMService().refine_lat(formatted_words, song_name)

    # key = a devanagari word , value = { "lat": latin_word, "lang": language_code }
    word_mapp = {}
    for word in refined_words:
        word_mapp[word["dev"]] = {"lat": word["lat"], "lang": word["lang"]}

    # word_mapping maps dev→lat
    utility.global_word_mapp = word_mapp

    return refined_words


def process_latin_script(lines: list[str], song_name: str) -> list[dict]:
    """
    Takes Hinglish/Latin lyrics lines, converts to Devanagari, refines via LLM and return it.
    """
    lyrics_text = " ".join(lines) 
    formatted_words = hinglish_to_devanagari(lyrics_text) # returns [{'lat':__, 'dev':__, 'lang':__}, ...]

    refined_words = LLMService().refine_dev(formatted_words, song_name)

    # key = a devanagari word , value = { "lat": latin_word, "lang": language_code }
    word_mapp = {}
    for word in refined_words:
        word_mapp[word["dev"]] = { "lat": word["lat"], "lang": word["lang"]}

    # word_mapping maps dev→lat
    utility.global_word_mapp = word_mapp

    return refined_words 