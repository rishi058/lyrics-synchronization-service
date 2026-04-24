import os
import time
from fastapi import APIRouter, HTTPException

from helpers.config import DEVICE, SUPPORTED_AUDIO_EXTS
from helpers.models import SyncLyricsRequest
from helpers.hi.transliteration import is_devanagari
from pre_processing import pre_process_audio

from pipeline._4_transcription import transcribe_chunk
from pipeline._5_alignment import align_chunk
from pipeline._6_timestamp_remapping import remap_timestamps
from pipeline._7_format_and_save import format_and_save

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/sync-lyrics")
def sync_lyrics(request: SyncLyricsRequest):
    start_time = time.time()
    print(f"[{time.strftime('%X')}] Using device: {DEVICE.upper()}")

    # ── Validation ────────────────────────────────────────────────────────────
    if not os.path.isfile(request.media_path):
        raise HTTPException(status_code=404, detail=f"Media file not found: {request.media_path}")

    _, ext = os.path.splitext(request.media_path)
    if ext.lower() not in SUPPORTED_AUDIO_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported format. Use {', '.join(SUPPORTED_AUDIO_EXTS)}")

    has_devanagari_lyrics = is_devanagari(request.lyrics)
    if request.language == "en" and request.lyrics and has_devanagari_lyrics:
        raise HTTPException(status_code=400, detail="Language is set to English but lyrics has Devanagari script.")

    if request.language == "en" and request.devanagari_output:  
        raise HTTPException(status_code=400, detail="Language is set to English but devanagari_output is True.")

    # ── PHASE 1: PRE-PROCESSING ────────────────────────
    segmented_data, duration = pre_process_audio(request.media_path)

    # ── PHASE 2: TRANSCRIPTION ────────────────────────
    segmented_data = transcribe_chunk(segmented_data, request.language, request.lyrics, request.media_path)  # adds "text" to each chunk
    print(f"[{time.strftime('%X')}] Transcription complete")

    
    # ── PHASE 3: ALIGNMENT ────────────────────────
    segmented_data = align_chunk(segmented_data, request.language)
    print(f"[{time.strftime('%X')}] Alignment complete")
    
    # ── PHASE 4: TIME REMAPPING ────────────────────────
    segmented_data = remap_timestamps(segmented_data)
    print(f"[{time.strftime('%X')}] Time remapping complete")

    # ── PHASE 5: FINAL OUTPUT ────────────────────────
    format_and_save(segmented_data, request.media_path, request.output_path, duration, request.language, request.devanagari_output)
    print(f"[{time.strftime('%X')}] Format and save complete")
    print(f"[{time.strftime('%X')}] Total time taken: {time.time() - start_time:.2f} seconds")

    return {"message": "Synchronization complete"}