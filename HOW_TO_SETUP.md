# 🛠️ Environment Setup Guide

This project uses **5 Python virtual environments** — one shared **base** and four **child** venvs.  
The base venv holds heavy packages (PyTorch CUDA, FastAPI, NumPy, etc.) and child venvs inherit them via a `.pth` symlink, so you don't waste ~15 GB duplicating PyTorch 4 times.

---

## Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| **Python** | **3.11.x** | `py -3.11 --version` |
| **pip** | any recent | comes with Python |
| **CUDA** | 12.8+ (for GPU) | `nvidia-smi` |
| **FFmpeg** | any recent | `ffmpeg -version` |

> If `py -3.11 --version` doesn't work, install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/).  
> Make sure to check **"Add to PATH"** during installation.

---

## Architecture Overview

```
d:\STUDY 2\test\
│
├── .venv\                              ← BASE venv (shared heavy packages)
│
├── app\
│   └── .venv\                          ← Child venv (app-specific packages)
│
├── transcribe\
│   ├── cohere-asr\
│   │   └── .venv\                      ← Child venv (cohere-specific packages)  
│   ├── qwen-asr\
│   │   └── .venv\                      ← Child venv (qwen-specific packages)
│   ├── whisperx\
│   │   └── .venv\                      ← Child venv (whisperx-specific packages)
│
├── requirements.txt                    ← Shared/base packages
└── HOW_TO_SETUP.md                     ← You are here
```

**How sharing works:**  
Each child venv has a tiny file called `_base_venv.pth` in its `Lib\site-packages\` folder. Python reads `.pth` files at startup and adds the listed paths to `sys.path`. This means every child can `import torch`, `import fastapi`, etc. without having its own copy.

---

## Step-by-Step Setup

> **All commands below are for PowerShell.** Open a terminal in VS Code (`Ctrl+`` `) or run PowerShell directly.  
> Always run from the project root

---

### Step 1 — Create & install the Base venv

The base venv holds: `torch` (CUDA 12.8), `torchvision`, `torchaudio`, `fastapi`, `uvicorn`, `numpy`, `librosa`, `soundfile`, `requests`, `python-dotenv`.

```powershell
# 1. Create the base venv
py -3.11 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. (Optional) Upgrade pip
python -m pip install --upgrade pip

# 4. Install shared packages
pip install -r requirements.txt
```

After this, verify:
```powershell
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: 2.8.0+cu128 CUDA: True
```

**Deactivate** when done (or just close the terminal):
```powershell
deactivate
```

---

### Step 2 — Create each Child venv

For every child, you do three things:

1. **Create** a plain venv
2. **Drop a `.pth` file** that points to the base venv's packages
3. **Install** only that module's own requirements

> ⚠️ **Important:** The `.pth` file must be saved as **UTF-8 without BOM**.  
> The PowerShell command below handles this correctly. Do NOT use `Out-File -Encoding utf8` — it adds a BOM that silently breaks Python.

---

#### 2a — App (`app\`, port 5000)

```powershell
# Create venv
py -3.11 -m venv app\.venv

# Create the .pth link to base packages
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$PWD\app\.venv\Lib\site-packages\_base_venv.pth", "$PWD\.venv\Lib\site-packages`n", $utf8)

# Activate child venv
.\app\.venv\Scripts\Activate.ps1

# Install app-specific packages
pip install -r app\requirements.txt

# Verify
python -c "import torch; print(torch.__version__); import demucs; print('demucs OK')"

deactivate
```

---

#### 2b — Cohere ASR (`transcribe\cohere-asr\`, port 5001)

```powershell
# Create venv
py -3.11 -m venv transcribe\cohere-asr\.venv

# Create the .pth link
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$PWD\transcribe\cohere-asr\.venv\Lib\site-packages\_base_venv.pth", "$PWD\.venv\Lib\site-packages`n", $utf8)

# Activate & install
.\transcribe\cohere-asr\.venv\Scripts\Activate.ps1
pip install -r transcribe\cohere-asr\requirements.txt

# Verify
python -c "import torch; print(torch.__version__); import transformers; print('transformers', transformers.__version__)"

deactivate
```

---

#### 2c — Qwen ASR (`transcribe\qwen-asr\`, port 5002)

```powershell
# Create venv
py -3.11 -m venv transcribe\qwen-asr\.venv

# Create the .pth link
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$PWD\transcribe\qwen-asr\.venv\Lib\site-packages\_base_venv.pth", "$PWD\.venv\Lib\site-packages`n", $utf8)

# Activate & install
.\transcribe\qwen-asr\.venv\Scripts\Activate.ps1
pip install -r transcribe\qwen-asr\requirements.txt

# Verify
python -c "import torch; print(torch.__version__); import qwen_asr; print('qwen_asr OK')"

deactivate
```

---

#### 2d — WhisperX (`transcribe\whisperx\`, port 5003)

```powershell
# Create venv
py -3.11 -m venv transcribe\whisperx\.venv

# Create the .pth link
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("$PWD\transcribe\whisperx\.venv\Lib\site-packages\_base_venv.pth", "$PWD\.venv\Lib\site-packages`n", $utf8)

# Activate & install
.\transcribe\whisperx\.venv\Scripts\Activate.ps1
pip install -r transcribe\whisperx\requirements.txt

# Verify
python -c "import torch; print(torch.__version__); import whisperx; print('whisperx OK')"

deactivate
```

---

## Running the Services

### Option A — VS Code Tasks (recommended)

1. Open VS Code
2. Press `Ctrl+Shift+P` → type **"Tasks: Run Task"**
3. Select **"⚡ Start All Services"**

This launches 4 terminals simultaneously:

| Terminal | Module | Port | Health Check |
|----------|--------|------|-------------|
| 🚀 App | `app\main.py` | 5000 | `http://localhost:5000/health` |
| 🎤 Cohere ASR | `transcribe\cohere-asr\main.py` | 5001 | `http://localhost:5001/health` |
| 🎤 Qwen ASR | `transcribe\qwen-asr\main.py` | 5002 | `http://localhost:5002/health` |
| 🎤 WhisperX | `transcribe\whisperx\main.py` | 5003 | `http://localhost:5003/health` |

You can also run individual services from the same task menu.

### Option B — Manual

Open separate terminals and run each.  
> **No need to activate the base venv first.** Each child venv's `python.exe` already knows where to find the base packages via the `_base_venv.pth` file — activation is only needed when you want to use bare `python` / `pip` commands.

```powershell
# Terminal 1 — App
.\app\.venv\Scripts\python.exe app\main.py

# Terminal 2 — Cohere ASR
.\transcribe\cohere-asr\.venv\Scripts\python.exe transcribe\cohere-asr\main.py

# Terminal 3 — Qwen ASR
.\transcribe\qwen-asr\.venv\Scripts\python.exe transcribe\qwen-asr\main.py

# Terminal 4 — WhisperX
.\transcribe\whisperx\.venv\Scripts\python.exe transcribe\whisperx\main.py

```

> **Tip:** You don't need to activate a venv to run its Python. Using the full path to `python.exe` inside the venv works the same way.

---

## Automated Setup (Optional)

If you trust the script and want to skip manual steps:

```powershell
# First-time setup
.\setup_venvs.ps1

# Rebuild everything from scratch
.\setup_venvs.ps1 -Force
```

---

## Troubleshooting

### `torch.cuda.is_available()` returns `False`
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure you installed the CUDA 12.8 build of torch (check `requirements.txt` has `+cu128`)
- If you're on CPU-only, torch will still work but inference will be slow

### Child venv can't find base packages (`ModuleNotFoundError`)
The `.pth` file is probably missing or corrupt. Recreate it:
```powershell
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText("<child_venv>\Lib\site-packages\_base_venv.pth", "<project_root>\.venv\Lib\site-packages`n", $utf8)
```
Replace `<child_venv>` and `<project_root>` with your actual paths.

### `pip install` re-installs torch in a child venv
Some packages (like `demucs`, `qwen-asr`) list `torch` as a dependency. pip will install it locally in the child even though the base already has it. This wastes space but doesn't break anything — the child's local copy will take priority. To clean it up:
```powershell
.\<child>\.venv\Scripts\pip.exe uninstall torch torchvision torchaudio -y
```

### PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### `py -3.11` not found
Use the full path instead:
```powershell
"C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```
