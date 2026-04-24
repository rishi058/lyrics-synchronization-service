import logging
import requests
import numpy as np
from typing import Literal
from fastapi import HTTPException

logger = logging.getLogger(__name__)

WHISPERX_BASE_URL = "http://localhost:5003/" 


def _serialize_segmented_data(segmented_data: list[dict]) -> list[dict]:
    """Convert np.ndarray audio fields to lists for JSON serialization."""
    return [
        {**chunk, "audio": chunk["audio"].tolist() if isinstance(chunk.get("audio"), np.ndarray) else chunk.get("audio")}
        for chunk in segmented_data
    ]

def _validate_response(data, url: str) -> list[dict]:
    """Ensure the alignment response is a list of dicts."""
    if not isinstance(data, list):
        raise ValueError(f"Alignment server at {url} returned {type(data).__name__}, expected list. Body: {str(data)[:500]}")
    return data

def align_chunk(segmented_data: list[dict], language: Literal["en", "hi"]) -> list[dict]:
    """
    Sends audio segments and text to the MFA alignment microservice.
    
    Returns:
    [
        {"start": 5.00, "end": 10.00, "aligned_words" : [
            {"word": ".......", "start": 0.00, "end": 5.00},
            {"word": ".......", "start": 5.00, "end": 10.00}, ...
        ]},
        ...
    ]
    """
    if not segmented_data:
        return []

    base_url = WHISPERX_BASE_URL

    url = base_url + ("align-hi" if language == "hi" else "align")
    
    try:
        res = requests.post(url, json=_serialize_segmented_data(segmented_data))
        res.raise_for_status()
        aligned_data = res.json()
        aligned_data = _validate_response(aligned_data, url)
        return aligned_data

    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to alignment service at {url}")
        raise HTTPException(status_code=503, detail="Alignment service is unreachable.")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout error from alignment service at {url}")
        raise HTTPException(status_code=504, detail="Alignment service request timed out.")
    except requests.exceptions.HTTPError as e:
        err_msg = e.response.text if e.response is not None else str(e)
        status_code = e.response.status_code if e.response is not None else 502
        logger.error(f"HTTP error {status_code} from alignment service at {url}: {err_msg}")
        raise HTTPException(status_code=502, detail=f"Alignment service failed with status {status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception to alignment service at {url}: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while contacting the alignment service.")