import requests

url = "http://localhost:5001/sync-lyrics"

if __name__ == "__main__":

    with open("lyrics.txt", "r", encoding="utf-8") as f:
        lyrics = f.read()

    body = {
        "media_path": "C:\\Users\\Rishi\\Downloads\\Afusic_-_Not_Enough.mp4",
        "output_path": "D:\\STUDY 2\\MediaEditor\\01\\test\\",
        "language": "hi",
        "lyrics": lyrics,
        "force_alignment": True,
        "devanagari_output": False,
        "isolate_vocals": True 
    }

    print("Sending request to the server...")
    response = requests.post(url, json=body)

    try:
        print(response.json())
    except Exception:
        print("Error:", response.status_code, response.text)

    # print("Request body:", body)