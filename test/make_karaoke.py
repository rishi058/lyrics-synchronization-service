import json, subprocess, io, os

# Minimum display time per word in centiseconds (prevents words from being skipped)
MIN_WORD_CS = 8  # 80ms minimum — enough for subtitle renderers to display

def fmt(ms):
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms_rem = divmod(rem, 1000)
    return f"{h}:{m:02}:{s:02}.{ms_rem//10:02}"

def make_karaoke(media, json_file, out_file):
    with open(json_file, encoding='utf-8') as f:
        words = json.load(f)

    # Skip blank/whitespace-only words (e.g. silence placeholders)
    words = [w for w in words if w['text'].strip()]

    # Alignment: 5 = Middle-Center (audio-only/MP3), 2 = Bottom-Center (MP4)
    align = "5" if media.endswith('.mp3') else "2"

    # ASS colour format: &HAABBGGRR  (AA=alpha, BB=blue, GG=green, RR=red)
    # White text primary, black outline/shadow for legibility
    # BackColour = black opaque background box
    PRIMARY   = "&H00FFFFFF"   # white
    SECONDARY = "&H0000FFFF"   # yellow fill (karaoke highlight)
    OUTLINE   = "&H00000000"   # black outline
    BACK      = "&H00000000"   # black background box
    # BorderStyle 4 = opaque box background; OutlineColour becomes the box colour in BS3/4
    # Use BorderStyle 3 (opaque box) with BackColour = black for solid bg behind text

    buf = io.StringIO()
    buf.write("[Script Info]\n")
    buf.write("ScriptType: v4.00+\n")
    buf.write("WrapStyle: 0\n")
    buf.write("ScaledBorderAndShadow: yes\n\n")
    buf.write("[V4+ Styles]\n")
    # Format fields: Name, Fontname, Fontsize,
    #   PrimaryColour (text), SecondaryColour (karaoke fill),
    #   OutlineColour, BackColour,
    #   Bold, Italic, Underline, StrikeOut,
    #   ScaleX, ScaleY, Spacing, Angle,
    #   BorderStyle, Outline, Shadow,
    #   Alignment, MarginL, MarginR, MarginV, Encoding
    buf.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
    buf.write(
        f"Style: K,Georgia,28,"          # Georgia is the closest freely-available serif to Amasis MT
        f"{PRIMARY},{SECONDARY},{OUTLINE},{BACK},"
        f"0,0,0,0,"                       # Bold/Italic/Underline/StrikeOut off
        f"100,100,0,0,"                   # ScaleX/Y, Spacing, Angle
        f"3,2,1,"                         # BorderStyle=3 (opaque box), Outline=2, Shadow=1
        f"{align},10,10,20,1\n\n"         # Alignment, margins, encoding
    )
    buf.write("[Events]\n")
    buf.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

    for w in words:
        raw_cs = (w['endMs'] - w['startMs']) // 10
        dur_cs = max(raw_cs, MIN_WORD_CS)   # enforce minimum so renderer never skips
        txt = w['text'].strip()
        buf.write(f"Dialogue: 0,{fmt(w['startMs'])},{fmt(w['endMs'])},K,,0,0,0,,{{\\K{dur_cs}}}{txt}\n")

    with open("sub.ass", "w", encoding='utf-8') as f:
        f.write(buf.getvalue())
    buf.close()

    vf = 'ass=sub.ass'
    if media.endswith('.mp3'):
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'color=c=black:s=1280x720:r=30',
            '-i', media,
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest', out_file
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-i', media,
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
            '-c:a', 'copy',
            out_file
        ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists("sub.ass"):
            os.remove("sub.ass")

if __name__ == "__main__":  
    # make_karaoke(
    #     'D:\\STUDY 2\\test\\test\\ip_media\\Afusic_Not_Enough.mp4', 
    #     'lyrics_json\\Afusic_Not_Enough.json',
    #     'op_media\\Afusic_Not_Enough_4200.mp4'
    # )
    make_karaoke(
        'D:\\STUDY 2\\test\\test\\ip_media\\01.mp4', 
        'lyrics_json\\01.json',
        'op_media\\01_lyrics.mp4'
    )   