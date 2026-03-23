import json, subprocess

def fmt(ms):
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms_rem = divmod(rem, 1000)
    return f"{h}:{m:02}:{s:02}.{ms_rem//10:02}"

def make_karaoke(media, json_file, out_file):
    with open(json_file, encoding='utf-8') as f: 
        words = json.load(f)
    
    # Alignment: 5 is Middle-Center (MP3), 2 is Bottom-Center (MP4)
    align = "5" if media.endswith('.mp3') else "2"
    
    ass = [
        "[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\n",
        "Format: Name, Fontname, Fontsize, PrimaryColour, Alignment\n",
        f"Style: K,Nirmala UI,36,&H00FFFF,{align}\n[Events]\n",
        "Format: Layer, Start, End, Style, Text\n"
    ]
    
    for w in words:
        dur_cs = (w['endMs'] - w['startMs']) // 10
        # Text is stripped to center perfectly without trailing spaces
        txt = f"{{\\K{dur_cs}}}{w['text'].strip()}"
        ass.append(f"Dialogue: 0,{fmt(w['startMs'])},{fmt(w['endMs'])},K,{txt}\n")

    with open("sub.ass", "w", encoding='utf-8') as f: 
        f.writelines(ass)

    vf = 'ass=sub.ass'
    if media.endswith('.mp3'):
        cmd = ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', media, '-vf', vf, '-shortest', out_file]
    else:
        cmd = ['ffmpeg', '-y', '-i', media, '-vf', vf, out_file]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    make_karaoke('C:\\Users\\Rishi\\Downloads\\Afusic_-_Not_Enough.mp4', 'Afusic_-_Not_Enough.json', 'not_enough_2.mp4')