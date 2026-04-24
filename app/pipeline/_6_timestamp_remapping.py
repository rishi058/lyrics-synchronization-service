"""
segmented_data:
[
    {"start": 5.00, "end": 10.00, "aligned_words" : [
        {"word": ".......", "start": 0.00, "end": 5.00},
        {"word": ".......", "start": 5.00, "end": 10.00}, ...
    ]},
    ...
]
"""

def remap_timestamps(segmented_data: list[dict]) -> list[dict]:
    """
    returns same format with updated timestamps
    """
    for segment in segmented_data:
        segment["aligned_words"] = _remap_timestamps_helper(segment["aligned_words"], segment["start"]) 

    segmented_data = _validate_word_time_span(segmented_data)

    return segmented_data


#!------------------PRIVATE HELPERS---------------------

def _remap_timestamps_helper(words_list: list[dict], original_start_sec: float = 0) -> list[dict]:
    offset = original_start_sec

    remapped = []
    for word in words_list:
        remapped.append({
                "word":       word.get("word", word.get("text", "")),
                "start":      word["start"]+ offset,
                "end":        word["end"]  + offset,
            })
    
    return remapped


def _validate_word_time_span(segmented_data: list[dict]) -> list[dict]:
    """
    If a word has a very short time-span (< 250ms), merge it with an adjacent word.
    RETURNS: same format as segmented_data
    """
    MIN_DURATION = 0.100  # 100 ms

    for segment in segmented_data:
        words = segment.get("aligned_words", [])
        if not words:
            continue

        merged_words = [words[0]]

        for current_word in words[1:]:
            duration = current_word["end"] - current_word["start"]

            if duration < MIN_DURATION:
                # Short word → absorb into previous
                prev_word = merged_words[-1]
                prev_word["word"] = f"{prev_word['word'].strip()} {current_word['word'].strip()}"
                prev_word["end"] = max(prev_word["end"], current_word["end"])
            else:
                merged_words.append(current_word)

        # If the first accumulated entry is still short, merge it forward
        if len(merged_words) > 1:
            first_duration = merged_words[0]["end"] - merged_words[0]["start"]
            if first_duration < MIN_DURATION:
                merged_words[1]["word"]  = f"{merged_words[0]['word'].strip()} {merged_words[1]['word'].strip()}"
                merged_words[1]["start"] = min(merged_words[0]["start"], merged_words[1]["start"])
                merged_words.pop(0)

        segment["aligned_words"] = merged_words

    return segmented_data