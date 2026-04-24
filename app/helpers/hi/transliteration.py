from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from helpers.config import DEVANAGARI_RE
from helpers.utils import _normalize_spaces
from wordfreq import zipf_frequency

def is_devanagari(text: str) -> bool:
    """Returns True if the text contains any Devanagari characters."""
    return bool(DEVANAGARI_RE.search(text))


def hinglish_to_devanagari(lyrics: str) -> list[dict]: 
    input_words = []

    lyrics = _normalize_spaces(lyrics)  # strip invisible Unicode spaces before splitting

    for word in lyrics.split():
        itrans = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
        lang = 'hi' if zipf_frequency(word, 'en') < 5 else 'en' 
        input_words.append({"lat": word, "dev": itrans, "lang": lang})

    return input_words  


def devanagari_to_hinglish(lyrics: str) -> list[dict]:
    input_words = []

    lyrics = _normalize_spaces(lyrics)  # strip invisible Unicode spaces before splitting

    for word in lyrics.split():
        lang = 'hi' if zipf_frequency(word, 'en') < 5 else 'en'

        if is_devanagari(word):
            # generate a hinglish word for the devanagari word
            hinglish = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
            input_words.append({"lat": hinglish, "dev": word, "lang": lang})
        else:
            itrans = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            input_words.append({"lat": word, "dev": itrans, "lang": lang})

    return input_words