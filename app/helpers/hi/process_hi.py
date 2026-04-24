from helpers.utils import clean_for_alignment
from llm.llm_service import LLMService
from helpers.hi.process_helper import process_devanagari_script, process_latin_script
from helpers.hi.transliteration import is_devanagari 
import helpers.utils as utility
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def process_hi_language(segment_data: list[dict], lyrics: str, song_name: str) -> list[dict]:
    """
    segment_data: flat list of {"start", "end", "audio", "text"} dicts.
    """
    transcribed_texts = [seg["text"] for seg in segment_data]   #! contains mixed scripts

    if lyrics:
        has_devanagari = is_devanagari(lyrics)  #! check the script of lyrics which user has provided

        #! Here considering lyrics provided by user doesn't contain any mixed scripts (OTHERWISE, CLEANING FUNCTION WILL REMOVE THE OTHER SCRIPT)
        if has_devanagari:
            lines = [clean_for_alignment(line, "devanagari") for line in lyrics.splitlines() if line.strip()]
            words_data = process_devanagari_script(lines, song_name)
        else:  # Latin script (Hinglish)
            lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
            words_data = process_latin_script(lines, song_name)

        #! NO MATTER WHAT SCRIPT USER PROVIDES, WE WILL CONVERT IT INTO LATIN(BCZ LLM EXPECTS LATIN SCRIPT)
        parts = [w["lat"] for w in words_data]
        lyrics_to_match = " ".join(parts).strip()

        #! ITERATE THROUGH TRANSCRIBED TEXT AND FORCE CONVERT to LATIN SCRIPT.
        for i, sentence in enumerate(transcribed_texts):
            words = sentence.split()
            for j, word in enumerate(words):
                if is_devanagari(word):
                    # first look in utility.global_word_mapp (key=dev, value={lat, lang})
                    entry = utility.global_word_mapp.get(word)
                    if entry is not None:
                        words[j] = entry["lat"]
                    else:
                        # if not found, force convert to hinglish via ITRANS
                        hinglish = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
                        words[j] = hinglish
            transcribed_texts[i] = " ".join(words)

        #! BEFORE FEEDING DATA TO LLM, NEED TO MAKE SURE TRANSCRIBED_TEXT SCRIPT AND LYRICS_TO_MATCH SCRIPT IS SAM(i.e LATIN)
        best_fit_lyrics = LLMService().refine_lyrics_segment(transcribed_texts, lyrics_to_match, "hi", song_name)

        #! BEFORE FEEDING DATA TO ALIGNER, NEED TO MAKE SURE SEGMENT TEXT ONLY INCLUDES DEV SCRIPT

        # build a reverse map: lat (lowercased) → dev, from global_word_mapp
        lat_to_dev = {v["lat"].lower(): k for k, v in utility.global_word_mapp.items()}

        for i, segment in enumerate(segment_data):
            lyric_line = best_fit_lyrics[i] if i < len(best_fit_lyrics) else segment["text"]
            words = lyric_line.split()
            for j, word in enumerate(words):
                # first look in reverse map (lat → dev)
                dev_word = lat_to_dev.get(word.lower())
                if dev_word is not None:
                    words[j] = dev_word
                else:
                    # fallback: force convert ITRANS → Devanagari
                    words[j] = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
            segment["text"] = " ".join(words)

    else:
        # No lyrics provided — correct word-by-word from Devanagari script
        words_data = process_devanagari_script(transcribed_texts, song_name)

        idx = 0
        for segment in segment_data:
            words_in_seg = segment["text"].split()
            corrected_words = []
            for _ in words_in_seg:
                if idx < len(words_data):
                    corrected_words.append(words_data[idx]["dev"])
                    idx += 1
            segment["text"] = " ".join(corrected_words)

    return segment_data
