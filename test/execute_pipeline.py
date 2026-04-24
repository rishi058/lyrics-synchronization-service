import requests

url = "http://localhost:5000/sync-lyrics"

if __name__ == "__main__":
    # with open("lyrics_text/die_for_you_lyrics.txt", "r", encoding="utf-8") as f:
    #     lyrics = f.read()

    # body = {
    #     "media_path": "D:\\STUDY 2\\test\\test\\ip_media\\Die_For_You.mp4",
    #     "output_path": "D:\\STUDY 2\\test\\test\\lyrics_json\\",
    #     "language": "en",
    #     "lyrics": lyrics,
    #     "devanagari_output": False,
    # }

    # with open("lyrics_text/afusic_not_enough_lyrics.txt", "r", encoding="utf-8") as f:
    #     lyrics = f.read()

    # body = {
    #     "media_path": "D:\\STUDY 2\\test\\test\\ip_media\\Afusic_Not_Enough.mp4",
    #     "output_path": "D:\\STUDY 2\\test\\test\\lyrics_json\\",
    #     "language": "hi",
    #     "lyrics": lyrics,
    #     "devanagari_output": False,
    # }

    body = {
        "media_path": "D:\\STUDY 2\\test\\test\\ip_media\\01.mp4",
        "output_path": "D:\\STUDY 2\\test\\test\\lyrics_json\\",
        "language": "hi",
        "lyrics": "",
        "devanagari_output": False,
    }
 
    print("Sending request to the server...")
    response = requests.post(url, json=body)

    try:
        print(response.json())
    except Exception:
        print("Error:", response.status_code, response.text)

    # print("Request body:", body)