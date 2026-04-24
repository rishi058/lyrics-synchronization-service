from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List
import numpy as np
import torch
import gc
import time

def _cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

DEVICE = "cuda"
SAMPLE_RATE = 16000
  
#! Need to check which model is best..  
# ALIGN_MODEL_EN = "facebook/wav2vec2-large-960h-lv60-self"
ALIGN_MODEL_EN = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

#! ONLY feasible working model for hindi 
ALIGN_MODEL_HI = "theainerd/Wav2Vec2-large-xlsr-hindi"  #! model size: 1.3 GB

app = FastAPI(
    title="WhisperX ASR Service",
    description="API for transcribing audio using WhisperX.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

class Segment(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    start: float
    end: float
    audio: Any  # np.ndarray — Pydantic v2 can't schema np.ndarray directly

class AlignSegment(Segment):
    text: str

#!-------------------------------------------------------------------------------------------------------------------------
# ── Model size reference ──────────────────────────────────────────────────────
# | Model            | Disk   | Mem      |
# |------------------|--------|----------|
# | tiny             | 75 MB  | ~390 MB  |
# | base             | 142 MB | ~500 MB  |
# | small            | 466 MB | ~1.0 GB  |
# | medium           | 1.5 GB | ~2.6 GB  |
# | large-v3         | 2.9 GB | ~4.7 GB  |
# | large-v3-turbo   | 1.5 GB | ~4.7 GB  |


import whisperx
   
def transcribe_helper(request: List[Segment], language: str):
    """
    Returns:
    [
        {"start": 5.00, "end": 10.00, "audio": np.ndarray, "text" : "____"},
        {"start": 14.00, "end": 19.00, "audio": np.ndarray, "text" : "____"},
        ...
    ]
    """
    print(f"[{time.strftime('%X')}] WHISPER TRANSCRIBE REQUEST RECEIVED")
    try:
        model = whisperx.load_model(
                "large-v3-turbo",
                DEVICE,
                compute_type="float16",
                asr_options={"beam_size": 6},
                language=language
            )
        print(f"[{time.strftime('%X')}] large-v3-turbo MODEL LOADED")
    except Exception as e:
        raise RuntimeError(f"Failed to load WhisperX ASR model: {e}") from e
    
    try:
        response_data = []

        for seg in request:
            try:
                audio_array = np.array(seg.audio, dtype=np.float32)  # JSON list → np.ndarray
                result = model.transcribe(
                    audio_array,
                    batch_size=6,
                    chunk_size=30,
                    language=language,
                    task="transcribe",
                )
            except Exception as e:
                raise RuntimeError(f"Failed to transcribe audio {seg.start} - {seg.end}: {e}") from e

            text = result.get("text") or " ".join(
                s.get("text", "") for s in result.get("segments", [])
            )
            response_data.append({"start": seg.start, "end": seg.end, "audio": seg.audio, "text": text.strip()})

        print(f"[{time.strftime('%X')}] TRANSCRIPTION COMPLETED")
        return response_data
    finally:
        del model
        _cleanup_gpu() 

@app.post("/transcribe")
def transcribe(request: List[Segment]):
    return transcribe_helper(request, "en")


@app.post("/transcribe-hi")
def transcribe_hi(request: List[Segment]):
    return transcribe_helper(request, "hi")

#!-------------------------------------------------------------------------------------------------------------------------

def align_helper(request: List[AlignSegment], language: str):
    """
    Returns:
    [
        {"start": 5.00, "end": 10.00, "aligned_words" : [
            {"word": ".......", "start": 5.00, "end": 6.00},
            {"word": ".......", "start": 6.00, "end": 10.00}, ...
        ]},
        ...
    ]
    """
    print(f"[{time.strftime('%X')}] WHISPER ALIGN REQUEST RECEIVED")
    try:
        model_name = ALIGN_MODEL_EN if language == "en" else ALIGN_MODEL_HI

        align_model, metadata = whisperx.load_align_model(
            language_code=language,
            device=DEVICE,
            model_name=model_name
        )

        print(f"[{time.strftime('%X')}] {model_name} MODEL LOADED")
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper Align model: {e}") from e

    try:
        response_data = []

        for seg in request:
            try:
                audio_array = np.array(seg.audio, dtype=np.float32)  # JSON list → np.ndarray

                duration = len(audio_array) / SAMPLE_RATE
                segment = [{"text": seg.text, "start": 0, "end": duration}]

                aligned = whisperx.align(
                    segment,
                    align_model,
                    metadata,
                    audio_array,
                    DEVICE,
                    return_char_alignments=False  # word-level is enough
                )

                # Extract word-level results from aligned output
                words = []
                for aligned_seg in aligned.get("segments", []):
                    words.extend(aligned_seg.get("words", []))   #{"word": ".......", "start": 0.00, "end": 5.00

            except Exception as e:
                raise RuntimeError(f"Failed to align audio {seg.start} - {seg.end}: {e}") from e

            response_data.append({
                "start": seg.start,
                "end": seg.end,
                "aligned_words": words
            })

        print(f"[{time.strftime('%X')}] ALIGNMENT COMPLETED")
        return response_data
    finally:
        del align_model, metadata
        _cleanup_gpu()

@app.post("/align")
def align(request: List[AlignSegment]):
    return align_helper(request, "en")

@app.post("/align-hi")
def align_hi(request: List[AlignSegment]):
    return align_helper(request, "hi")

#!-------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5003)     


#!-------------------------------------------------------------------------------------------------------------------------
#! IF WANT TO TEST THIS MODULE INDEPENDENTLY:

# import whisperx

# if __name__=="__main__":
#     try:
#         model = whisperx.load_model(
#                 "large-v3-turbo", "cuda",
#                 compute_type="float16",
#                 asr_options={"beam_size": 6},
#                 language="en"
#             )
    
    
#         audio="D:\\STUDY 2\\test\\app\\cache\\seperations\\Die_For_You_vocals.wav"

#         result = model.transcribe(
#             audio,
#             batch_size=6,         # increase for GPU
#             chunk_size=30,
#             language="en",
#             task="transcribe",     # not translate
#         )

#         # result is { "text": str, "segments": list, "language": str }
#         # When no speech is detected, WhisperX omits the top-level "text" key.

#         text = result.get("text") or " ".join(
#             seg.get("text", "") for seg in result.get("segments", [])
#         )

#         # write text in output.txt
#         with open("output.txt", "w") as f:
#             f.write(text.strip())

#     except Exception as e:
#         raise RuntimeError(f"Transcription failed: {e}") from e

   