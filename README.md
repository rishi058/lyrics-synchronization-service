# Lyrics Synchronization Service

Welcome to the Lyrics Synchronization Service! This tool is designed to take an audio or video file along with/without its lyrics and output a time-synchronized JSON file containing word-by-word timing data.

Recommended to provide High Quality audio/video and lyrics for better accuracy.
---

## 🛠️ Setup Instructions

### Requirements
Before getting started, please ensure you have the following installed on your system:
- **FFmpeg**: Must be installed and added to your system's `PATH`.
- **NVIDIA GPU**: Required for CUDA support to run the models efficiently.
- **GEMINI API KEY**: Required for a better accurate result to refine the output.

### Installation
1. Clone the repository to your local machine:
   ```bash
   git clone <repository-url>
   ```
2. Navigate into the project directory:
   ```bash
   cd lyrics-synchronization-service
   ```
3. Follow the detailed environment setup instructions provided in:
   👉 **[`HOW_TO_SETUP.md`](HOW_TO_SETUP.md)**

> **Note for Developers**: If you are a developer looking to understand the underlying architecture, model pipeline, and technical decisions, please explore **[`DEV_README.md`](DEV_README.md)**.

---

## ⚙️ High-Level Overview

Once you have completed the setup and started the services, the system runs **4 distinct virtual environments/projects** simultaneously. These services consume the following 4 local ports:
- **`5000`** - Main App Pipeline (API Gateway)
- **`5001`** - Cohere ASR Service
- **`5002`** - Qwen ASR Service
- **`5003`** - WhisperX Service

---

## 📡 How to Consume (For End-Users)

To use the synchronization service, you need to make an HTTP `POST` request to the main application running on port `5000`.

> **NOTE** : For first run it will take time to download the models. It will take 10-15 mins depending on your internet speed.

**Endpoint:**
`POST http://localhost:5000/sync-lyrics`

**Request Body (JSON):**
```json
{
    "media_path": "C:/absolute/path/to/your/audio_or_video_file.mp3",
    "output_path": "C:/absolute/path/to/save/synchronized_output.json",
    "language": "en", (Only two language is supported : "en" for English and "hi" for Hindi)
    "lyrics": "Your raw lyrics text goes here...", (leave empty for auto-lyrics - not recommended)
    "devanagari_output": false
}
```

- Set `devanagari_output` to `true` or `false` based on your needs for Hindi.*
   
### 📄 Expected Output

After the synchronization process finishes, the service will generate and save a JSON file at your specified `output_path`. The generated JSON will have the following structure:

```json
[
    {
        "text": " Oh",
        "startMs": 27880,
        "endMs": 27960,
        "timestampMs": 27880
    },
    {
        "text": " yeah",
        "startMs": 27970,
        "endMs": 28500,
        "timestampMs": 27970
    }
]
```

