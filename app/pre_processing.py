import soundfile as sf
import time
from pipeline._1_ingestion import ingest
from pipeline._2_seperation import separate_vocals
from pipeline._3_vad_chunking import vad_chunking

def pre_process_audio(media_path: str) -> tuple[list[dict], float]:
    formatted_media_path = ingest(media_path)   # removes video, forces 16k Hz sample rate
    print(f"[{time.strftime('%X')}] Ingestion complete")

    vocal_media_path = separate_vocals(formatted_media_path)
    print(f"[{time.strftime('%X')}] Vocal separation complete")

    segmented_data = vad_chunking(vocal_media_path)  # returns [{"start": 0.00, "end": 10.00, "audio": np.ndarray},...] 
    print(f"[{time.strftime('%X')}] VAD chunking complete")

    data, sr = sf.read(vocal_media_path)
    duration = len(data) / sr
 
    return segmented_data, duration