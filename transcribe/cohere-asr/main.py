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
    title="Cohere ASR Service",
    description="API for transcribing audio using Cohere ASR.",
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

#!-------------------------------------------------------------------------------------------------------------------------

from transformers import AutoProcessor, CohereAsrForConditionalGeneration 
# Download ~5GB model in C:\Users\<username>\.cache\huggingface\hub\models--CohereLabs--cohere-transcribe-03-2026
   
@app.post("/transcribe")
def transcribe(request: List[Segment]):
    """
    Returns:
    [
        {"start": 5.00, "end": 10.00, "audio": np.ndarray, "text" : "____"},
        {"start": 14.00, "end": 19.00, "audio": np.ndarray, "text" : "____"},
        ...
    ]
    """
    print(f"[{time.strftime('%X')}] COHERE TRANSCRIBE REQUEST RECEIVED")
    try:
        model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto") 
        processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
        print(f"[{time.strftime('%X')}] cohere-transcribe-03-2026 MODEL LOADED")
    except Exception as e:
        raise RuntimeError(f"Failed to load Cohere ASR model: {e}") from e
    
    try:
        response_data = []

        for seg in request:
            try:
                # seg.audio arrives as a plain Python list (JSON-deserialized)
                audio_array = np.array(seg.audio, dtype=np.float32)
                inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", language="en")
                inputs = inputs.to(model.device, dtype=model.dtype)  # must re-assign; .to() returns new object
                outputs = model.generate(**inputs, max_new_tokens=256)
                text = processor.batch_decode(outputs, skip_special_tokens=True)[0]  # outputs is batched
            except Exception as e:
                raise RuntimeError(f"Failed to transcribe audio {seg.start} - {seg.end}: {e}") from e

            response_data.append({"start": seg.start, "end": seg.end, "audio": seg.audio, "text": text.strip()})

        print(f"[{time.strftime('%X')}] TRANSCRIPTION COMPLETED")
        return response_data
    finally:
        del model, processor
        _cleanup_gpu() 

#!-------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5001)     


#!-------------------------------------------------------------------------------------------------------------------------
#! IF WANT TO TEST THIS MODULE INDEPENDENTLY:

# from transformers import AutoProcessor, CohereAsrForConditionalGeneration
# from transformers.audio_utils import load_audio

# if __name__ == "__main__": 
#     model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto") 
#     processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    
#     audio_file_path = "D:\\STUDY 2\\test\\app\\cache\\seperations\\Die_For_You_vocals.wav"

#     audio = load_audio(audio_file_path, sampling_rate=16000)

#     inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
#     inputs.to(model.device, dtype=model.dtype)

#     outputs = model.generate(**inputs, max_new_tokens=256)
#     text = processor.decode(outputs, skip_special_tokens=True)
    
#     # write text in output.txt
#     with open("output.txt", "w") as f:
#         f.write(str(text))

#     print("Transcription completed successfully!")