import whisperx
import time
import gc
import torch
from helpers.config import COMPUTE_TYPE, DEVICE, MODEL_NAME, ALIGN_MODEL_EN, SAMPLE_RATE
from helpers.utils import clean_for_alignment
from helpers.silero_vad import detect_vocal_bounds, get_speech_chunks, merge_overlapping_words

from helpers.logger import CustomLogger

def process_english_language(lyrics: str, audio) -> list[dict]:
    """Returns sync data (list of aligned word dicts) for English language."""

    audio_duration = len(audio) / SAMPLE_RATE
    vocal_start, vocal_end = detect_vocal_bounds(audio, audio_duration)

    if not lyrics:
        # Transcribe and align using VAD chunks
        print(f"[{time.strftime('%X')}] Transcribing and aligning English words with VAD chunking...")
        try:
            model = whisperx.load_model(MODEL_NAME, DEVICE, compute_type=COMPUTE_TYPE, asr_options={"beam_size": 5}, language="en")

            model_a, metadata = whisperx.load_align_model(
                language_code="en", device=DEVICE, model_name=ALIGN_MODEL_EN
            )
            
            chunks = get_speech_chunks(audio)
            
            global_aligned_words = []
            
            for chunk in chunks:
                chunk_start = chunk["start"]
                chunk_end = min(chunk["end"], audio_duration)
                
                start_sample = int(chunk_start * SAMPLE_RATE)
                end_sample = int(chunk_end * SAMPLE_RATE)
                audio_chunk = audio[start_sample:end_sample]
                
                if len(audio_chunk) == 0:
                    continue
                    
                # 1. WhisperX Local Transcription
                result = model.transcribe(audio_chunk, batch_size=5, chunk_size=10, language="en")
                chunk_segments = result.get("segments", [])
                
                if not chunk_segments:
                    continue
                    
                # 2. WhisperX Local Alignment
                result_aligned = whisperx.align(chunk_segments, model_a, metadata, audio_chunk, DEVICE)
                
                # 3. Timeline Mapping
                for word_data in result_aligned.get("word_segments", []):
                    if "start" in word_data and "end" in word_data:
                        global_word = {
                            "word": word_data["word"],
                            "start": round(chunk_start + word_data["start"], 3),
                            "end": round(chunk_start + word_data["end"], 3)
                        }
                        if "score" in word_data:
                            global_word["score"] = word_data["score"]
                        global_aligned_words.append(global_word)


                # Release VRAM 
                del audio_chunk, result, chunk_segments, result_aligned
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
            aligned_words = global_aligned_words

        except Exception as e:
            raise RuntimeError(f"Transcription/Alignment failed: {e}") from e

        if not aligned_words:
            raise RuntimeError("English alignment produced no words.")
            
        return merge_overlapping_words(aligned_words)

    else:
        # Use provided lyrics as segments
        lines = [clean_for_alignment(line, "latin") for line in lyrics.splitlines() if line.strip()]
        segments = [{"text": " ".join(lines), "start": vocal_start, "end": vocal_end}]

        # Align the transcript with the audio
        try:
            print(f"[{time.strftime('%X')}] Aligning English words...")
            model_a, metadata = whisperx.load_align_model(
                language_code="en", device=DEVICE, model_name=ALIGN_MODEL_EN
            )
            result_aligned = whisperx.align(segments, model_a, metadata, audio, DEVICE)
            aligned_words = result_aligned["word_segments"]
        except Exception as e:
            raise RuntimeError(f"English alignment failed: {e}") from e

        if not aligned_words:
            raise RuntimeError("English alignment produced no words.")

        return aligned_words
