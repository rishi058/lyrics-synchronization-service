from typing import Literal
import logging
import numpy as np
import requests
from fastapi import HTTPException
from helpers.en.process_en import process_en_language
from helpers.hi.process_hi import process_hi_language
import os
import json

logger = logging.getLogger(__name__)

COHERE_BASE_URL = "http://localhost:5001/"
QWEN_BASE_URL = "http://localhost:5002/"
WHISPERX_BASE_URL = "http://localhost:5003/"

def transcribe_chunk(segmented_data: list[dict], language: Literal["en", "hi"], lyrics: str, media_path: str) -> list[dict]:
    """
    Returns:
    [
        {"start": 5.00, "end": 10.00, "audio": np.ndarray, "text" : "____"},
        {"start": 14.00, "end": 19.00, "audio": np.ndarray, "text" : "____"},
        ...
    ]
    """
    # ── Path setup and caching ──────────────────────────────────────────────
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    song_name = os.path.splitext(os.path.basename(media_path))[0]
    cache_dir = os.path.join(root_dir, "cache", "transcriptions")
    os.makedirs(cache_dir, exist_ok=True)
    cache_json = os.path.join(cache_dir, f"{song_name}_{language}_raw_transcription.json")
    cache_npz  = os.path.join(cache_dir, f"{song_name}_{language}_raw_transcription_audio.npz")

    if os.path.exists(cache_json) and os.path.exists(cache_npz):
        logger.info(f"Loading cached raw transcriptions from {cache_json}")
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        audio_store = np.load(cache_npz)
        segmented_data = [
            {**seg, "audio": audio_store[str(i)]}
            for i, seg in enumerate(meta)
        ]
        
    else:
        # Capture audio BEFORE the API call — the response comes back as plain lists (JSON round-trip), so isinstance(audio, np.ndarray) would be
        original_audio = {str(i): seg["audio"] for i, seg in enumerate(segmented_data) if isinstance(seg.get("audio"), np.ndarray)}

        #!----------------------------------------------------------------------------------------------------------------------
        if language == "en":
            # url = COHERE_BASE_URL + "transcribe"
            url = QWEN_BASE_URL + "transcribe"          
            # url = WHISPERX_BASE_URL + "transcribe"
        elif language == "hi":
            # url = QWEN_BASE_URL + "transcribe-hi"  
            url = WHISPERX_BASE_URL + "transcribe-hi"     #! Note: may return mixed script (dev+latin) 
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")

        #!----------------------------------------------------------------------------------------------------------------------
        
        try:
            res = requests.post(url, json=_serialize_segmented_data(segmented_data))
            res.raise_for_status()
            raw = res.json()
            segmented_data = _validate_response(raw, url)
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to transcription service at {url}")  
            raise HTTPException(status_code=503, detail="Transcription service is unreachable.")
        except requests.exceptions.Timeout:
            logger.error(f"Timeout error from transcription service at {url}")
            raise HTTPException(status_code=504, detail="Transcription service request timed out.")
        except requests.exceptions.HTTPError as e:
            err_msg = e.response.text if e.response is not None else str(e)
            status_code = e.response.status_code if e.response is not None else 502
            logger.error(f"HTTP error {status_code} from transcription service at {url}: {err_msg}")
            raise HTTPException(status_code=502, detail=f"Transcription service failed with status {status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception to transcription service at {url}: {str(e)}")
            raise HTTPException(status_code=500, detail="An error occurred while contacting the transcription service.")

        #!----------------------------------------------------------------------------------------------------------------------

        # Restore original np.ndarray audio onto the response segments
        for i, seg in enumerate(segmented_data):
            if str(i) in original_audio:
                seg["audio"] = original_audio[str(i)]

        # Save metadata+text (human-editable) and audio arrays separately
        meta = [{k: v for k, v in seg.items() if k != "audio"} for seg in segmented_data]
        with open(cache_json, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
        np.savez(cache_npz, **original_audio)
        logger.info(f"Transcription cache saved → {cache_json} + {cache_npz}")

        #!----------------------------------------------------------------------------------------------------------------------

    if language == "hi":
        segmented_data = process_hi_language(segmented_data, lyrics, song_name)
    elif language == "en":
        # If LYRICS are provided, Refines the text of the segmented data to match it
        segmented_data = process_en_language(segmented_data, lyrics, song_name)


    return segmented_data

#!--------------------------- PRIVATE HELPERS -----------------------------------------

def _serialize_segmented_data(segmented_data: list[dict]) -> list[dict]:
    """Convert np.ndarray audio fields to lists for JSON serialization."""
    return [
        {**chunk, "audio": chunk["audio"].tolist() if isinstance(chunk.get("audio"), np.ndarray) else chunk.get("audio")}
        for chunk in segmented_data
    ]

def _validate_response(data, url: str) -> list[dict]:
    """Ensure the transcription response is a list of dicts with a 'text' key."""
    if not isinstance(data, list):
        raise ValueError(f"Transcription server at {url} returned {type(data).__name__}, expected list. Body: {str(data)[:500]}")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Transcription server at {url}: item[{i}] is {type(item).__name__} (value={repr(item)[:200]}), "
                f"expected dict with keys 'start','end','audio','text'."
            )
        if "text" not in item:
            raise ValueError(f"Transcription server at {url}: item[{i}] has no 'text' key. Keys present: {list(item.keys())}")
    return data