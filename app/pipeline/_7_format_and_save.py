import os
import json
from helpers.utils import format_segment_for_hindi

def format_and_save(segmented_data: list[dict], media_path: str, output_dir: str, audio_duration: float, language: str, devanagari_output: bool) -> None:
    
    if language == "hi":
        segmented_data = format_segment_for_hindi(segmented_data, devanagari_output)
    
    media_name = os.path.splitext(os.path.basename(media_path))[0] 

    save_data(segmented_data, f"{media_name}_raw.json", output_dir)
            
    # FORMAT DATA — iterate over words within each chunk for word-level output
    final_data = []
    for seg in segmented_data:
        for word in seg["aligned_words"]:
            final_data.append({
                "text": " " + word["word"],  
                "startMs": int(word["start"] * 1000),
                "endMs": int(word["end"] * 1000),
                "timestampMs": int(word["start"] * 1000),
            })


    final_data.insert(0, {
        "text": " ", "startMs": 0,
        "endMs": max(0, final_data[0]["startMs"] - 1),
        "timestampMs": 0
    })

    final_data.append({
        "text": " ", "startMs": final_data[-1]["endMs"] + 1,
        "endMs": int(audio_duration * 1000),
        "timestampMs": final_data[-1]["endMs"] + 1
    })

    save_data(final_data, f"{media_name}.json", output_dir)

def save_data(data: list[dict], file_name: str, output_dir: str) -> None:
    output_path = os.path.join(output_dir, file_name)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)