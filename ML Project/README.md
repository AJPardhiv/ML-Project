# Gesture-Based Laptop Controller

This project enables laptop interaction using:
- hand gestures from webcam input
- optional offline voice commands
- optional ML-based gesture classification

All core processing is local (offline).

## Features

- Real-time gesture-based mouse control
- Click and scroll gestures
- Optional voice command integration using Vosk + pyttsx3
- ML data collection, model training, and live inference pipeline

## Requirements

- Python 3.8+
- Webcam
- Microphone (only for voice mode)
- Windows/macOS/Linux

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup_check.py
```

For voice mode, keep a valid Vosk model inside the `model/` directory.

## Run

### Main application

```bash
python main.py
```

Useful options:

```bash
python main.py --no-voice
python main.py --voice-model model
python main.py --mic-device 0
python main.py --mic-samplerate 16000
```

### ML workflow

1) Collect training data:

```bash
python collect_data.py --output gesture_data.csv --samples 50
```

2) Train model:

```bash
python train_model.py --data gesture_data.csv --type rf --output models/gesture_model.joblib
```

3) Live inference:

```bash
python infer_live.py --model models/gesture_model.joblib --threshold 0.6
```

Optional action execution in inference mode:

```bash
python infer_live.py --model models/gesture_model.joblib --with-actions
```

## Project Structure

- `main.py`: application entry point
- `gesture_controller.py`: gesture detection logic
- `voice_assistant.py`: voice command processing
- `actions.py`: action data model and queue
- `collect_data.py`: labeled data collection
- `train_model.py`: model training and evaluation
- `infer_live.py`: real-time inference
- `setup_check.py`: environment/device validation

## Troubleshooting

- Webcam unavailable: close other apps that use the camera.
- Voice unavailable: verify microphone permissions and `model/` contents.
- Desktop actions blocked on Windows: run terminal as administrator.
- Poor gesture detection: improve lighting and reduce background clutter.

## Evaluation Notes

Suggested demo order:
1. `python setup_check.py`
2. `python main.py --no-voice`
3. show move, click, scroll, pause
4. optionally run voice mode (`python main.py`)
5. optionally run ML inference (`python infer_live.py --model models/gesture_model.joblib`)
