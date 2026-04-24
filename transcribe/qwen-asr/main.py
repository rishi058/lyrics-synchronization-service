from torch.cuda import temperature
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

app = FastAPI(
    title="Qwen ASR Service",
    description="API for transcribing audio using Qwen ASR.",
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

from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
 
# Needed to handle mixed script
CONTEXT_FOR_HI = "DO NOT TRANSLATE or SKIP any words, It may contain English words at(START, MIDDLE, END) transcribe them also."

# Just added to ensure safety
CONTEXT_FOR_EN = "DO NOT SKIP any words escpecially at START and END of the audio."

#! WILL DOWNLOAD MODEL ONCE OF SIZE 5 GB
def transcribe_helper(request: List[Segment], language:str):
    """
    Returns:
    [
        {"start": 5.00, "end": 10.00, "audio": np.ndarray, "text" : "____"},
        {"start": 14.00, "end": 19.00, "audio": np.ndarray, "text" : "____"},
        ...
    ]
    """
    print(f"[{time.strftime('%X')}] QWEN TRANSCRIBE REQUEST RECEIVED")
    try:
        model = Qwen3ASRModel.from_pretrained(
                    "Qwen/Qwen3-ASR-1.7B",
                    dtype=torch.bfloat16, 
                    device_map="cuda:0",
                    max_inference_batch_size=1, 
                    max_new_tokens=512,
                )
        print(f"[{time.strftime('%X')}] Qwen3-ASR-1.7B MODEL LOADED")
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen ASR model: {e}") from e
    
    try:
        response_data = []

        for seg in request:
            try:
                audio_array = np.array(seg.audio, dtype=np.float32)  # JSON list → np.ndarray

                context = CONTEXT_FOR_HI if language == "Hindi" else CONTEXT_FOR_EN

                asr_results = model.transcribe(
                    audio=(audio_array, 16000),
                    language=language,               #! None = auto LID, or pass "Hindi"/"English" to force
                    context = context,
                    return_time_stamps=False,
                )

            except Exception as e:
                raise RuntimeError(f"Failed to transcribe audio {seg.start} - {seg.end}: {e}") from e

            text = asr_results[0].text
            response_data.append({"start": seg.start, "end": seg.end, "audio": seg.audio, "text": text.strip()})

        print(f"[{time.strftime('%X')}] TRANSCRIPTION COMPLETED")
        return response_data
    finally:
        del model
        _cleanup_gpu()

@app.post("/transcribe")
def transcribe(request: List[Segment]):
    return transcribe_helper(request, "English")


@app.post("/transcribe-hi")
def transcribe_hi(request: List[Segment]):
    return transcribe_helper(request, "Hindi")

#!-------------------------------------------------------------------------------------------------------------------------

#! NOTE: QWEN ONLY PROVIDES FORCED ALIGNMENT FOR ENGLISH
@app.post("/align")
def align(request: List[AlignSegment]):
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
    print(f"[{time.strftime('%X')}] QWEN ALIGN REQUEST RECEIVED")
    try:
        model = Qwen3ForcedAligner.from_pretrained(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        print(f"[{time.strftime('%X')}] Qwen3-ForcedAligner-0.6B MODEL LOADED")
    except Exception as e:
        raise RuntimeError(f"Failed to load Qwen3-ForcedAligner model: {e}") from e

    try:
        response_data = []

        for seg in request:
            try:
                audio_array = np.array(seg.audio, dtype=np.float32)  # JSON list → np.ndarray
                
                align_out = model.align(
                    audio=(audio_array, 16000),
                    text=seg.text,
                    language="English",
                )
            except Exception as e:
                raise RuntimeError(f"Failed to align audio {seg.start} - {seg.end}: {e}") from e

            aligned_words = []
            if align_out:
                for word_obj in align_out[0]:
                    aligned_words.append({
                        "word": word_obj.text,
                        "start": round(word_obj.start_time, 2),
                        "end": round(word_obj.end_time, 2)
                    })

            response_data.append({
                "start": seg.start,
                "end": seg.end,
                "aligned_words": aligned_words
            })

        print(f"[{time.strftime('%X')}] ALIGNMENT COMPLETED")
        return response_data
    finally:
        del model
        _cleanup_gpu()

#!-------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5002)     


#!-------------------------------------------------------------------------------------------------------------------------
#! IF WANT TO TEST THIS MODULE INDEPENDENTLY:

# from qwen_asr import Qwen3ASRModel
# import torch 
   
# if __name__ == "__main__":
#     asr_model = Qwen3ASRModel.from_pretrained(
#                 "Qwen/Qwen3-ASR-1.7B",
#                 dtype=torch.bfloat16, 
#                 device_map="cuda:0",
#                 max_inference_batch_size=1, 
#                 max_new_tokens=512,
#             ) 

#     asr_results = asr_model.transcribe(
#         audio="D:\\STUDY 2\\test\\app\\cache\\seperations\\Die_For_You_vocals.wav",
#         language="English",
#         return_time_stamps=False, # Wait to align until next step 
#     )

#     text = asr_results[0].text

#     # write text in output.txt
#     with open("output.txt", "w") as f:
#         f.write(text)

#     print("Transcription completed successfully!")


