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

### 2. Tic-Tac-Toe AI

Unbeatable AI using Minimax + Alpha-Beta Pruning.

File: tic_tac_toe_ai.py
Features:
Player vs AI
AI never loses
Input validation

Run:

python tic_tac_toe_ai.py
3. Image Captioning AI

Generates captions using CNN + LSTM/Transformer.

Features:
ResNet50 / VGG16 encoder
LSTM / Transformer decoder
Custom dataset support

Train:

python train.py

Inference:

python inference.py --image_path data/images/sample.jpg --checkpoint artifacts/model.pt --vocab_path artifacts/vocab.json --encoder resnet50 --decoder lstm
