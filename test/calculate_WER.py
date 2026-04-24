import re

def normalize(text):
    text = text.lower()
    text = re.sub(r"[\"',.()!?]", "", text)
    return text.split()

def wer(ref, hyp):
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(ref)][len(hyp)] / len(ref)

with open(r"d:\STUDY 2\test\app\test\lyrics_text\lyrics.txt", "r", encoding="utf-8") as f:
    lyrics = normalize(f.read())
with open(r"d:\STUDY 2\test\app\test\output_1.txt", "r", encoding="utf-8") as f:
    out1 = normalize(f.read())
with open(r"d:\STUDY 2\test\cohere-command\output.txt", "r", encoding="utf-8") as f:
    out2 = normalize(f.read())

print(f"Length Lyrics: {len(lyrics)}, Output 1: {len(out1)}, Output 2: {len(out2)}")
print(f"Output 1 WER (Lower is better): {wer(lyrics, out1):.2%}")
print(f"Output 2 WER (Lower is better): {wer(lyrics, out2):.2%}")
