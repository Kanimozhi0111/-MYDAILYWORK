<<<<<<< HEAD
# Face detection and recognition (Task 5)

Small AI demo that finds faces in **images** or **videos** and can optionally **name** people you enroll locally.

## What is inside

| Piece | Role |
|--------|------|
| **Haar cascades** (OpenCV) | Classic, fast frontal-face detector. |
| **MediaPipe Face Detection** | Lightweight neural detector (good default). |
| **LBPH recognizer** (OpenCV `cv2.face`) | Optional recognition on your own photo sets. |

For coursework that mentions **Siamese networks** or **ArcFace**: those learn embedding spaces for large-scale identity; this project uses **LBPH** so everything runs locally without GPU-only stacks. You can extend the same UI later by swapping the recognizer for DeepFace, InsightFace, or a custom ArcFace ONNX model.

## Setup (Windows)

Use a virtual environment inside this folder (recommended).

```powershell
cd C:\Users\Kanimozhi\MYDAILYWORK\face_detection_recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use **`opencv-contrib-python`** only (not plain `opencv-python`), because **LBPH** lives in the `contrib` package.

On first run with **MediaPipe**, the app downloads a small BlazeFace model into `models/` (required for MediaPipe 0.10.31+, which no longer includes `mp.solutions`).

## Run the app

```powershell
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Enroll people for recognition

1. Under `data\enrolled\`, create one folder per person, for example `data\enrolled\Alice\` and `data\enrolled\Bob\`.
2. Add a few **frontal** face photos per person (`.jpg`, `.png`, etc.).
3. In the sidebar, turn on **Label enrolled faces** and reload the app if needed.

Training scans enrollment images with the **Haar** detector to crop faces, then trains LBPH. If recognition is wrong, add more varied photos or adjust **LBPH distance threshold** (lower = stricter).

## Project layout

```
face_detection_recognition/
  app.py            # Streamlit UI
  detection.py      # Haar + MediaPipe detection helpers
  recognition.py    # LBPH training / prediction
  data/
    enrolled/       # <Name>/photos…
  requirements.txt
```

## Privacy note

All processing is local. Do not commit private biometric data; keep enrollment photos out of git if they are sensitive.
=======
# MYDAILYWORK – AI & Python Projects Collection

This repository contains multiple Python-based projects demonstrating rule-based systems, game AI, and deep learning.

---

## Projects

### 1. Rule-Based Chatbot
A simple command-line chatbot using Python and regex.

- File: `rule_based_chatbot.py`
- Features:
  - Pattern-based responses
  - Multiple intents (greetings, help, motivation, etc.)
  - Exit commands supported

Run:
```bash
python rule_based_chatbot.py
```

---

### 2. Tic-Tac-Toe AI

Unbeatable AI using Minimax + Alpha-Beta Pruning.

File: tic_tac_toe_ai.py
Features:
Player vs AI
AI never loses
Input validation

Run:
```bash
python tic_tac_toe_ai.py
```

---

### 3. Image Captioning AI

Generates captions using CNN + LSTM/Transformer.

Features:
ResNet50 / VGG16 encoder
LSTM / Transformer decoder
Custom dataset support

Requirements:
```bash
pip install -r requirements.txt
```
Train:
```bash
python train.py
```
Inference:
```bash
python inference.py --image_path data/images/sample.jpg --checkpoint artifacts/model.pt --vocab_path artifacts/vocab.json --encoder resnet50 --decoder lstm
```
>>>>>>> 76aa1dab08024eea4b789e18a993b69817f37df2
