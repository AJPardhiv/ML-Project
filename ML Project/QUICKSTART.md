# Quickstart

## 1. Basic Run (No ML Training)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup_check.py
python main.py
```

This mode gives rule-based gesture control immediately.

## 2. Voice Mode (Optional)

1. Ensure a Vosk model exists in the `model` folder.
2. Run:

```bash
python main.py
```

If voice is not needed:

```bash
python main.py --no-voice
```

## 3. ML Pipeline (Optional)

### 3.1 Collect Data

```bash
python collect_data.py --output gesture_data.csv --samples 50
```

Labels:

- `1` move
- `2` click
- `3` scroll
- `4` pause
- `Q` quit

### 3.2 Train Model

```bash
python train_model.py --data gesture_data.csv --type rf --output models/gesture_model.joblib
```

### 3.3 Run Live Inference

```bash
python infer_live.py --model models/gesture_model.joblib --threshold 0.6
```

To allow click/scroll execution during inference:

```bash
python infer_live.py --model models/gesture_model.joblib --with-actions
```

## 4. Common Issues

- Webcam unavailable: close other camera apps.
- Voice not detected: verify microphone permissions and `model` folder.
- Action blocked on Windows: run terminal as administrator.

## 5. Minimal Demo Script for Evaluation

```bash
python setup_check.py
python main.py --no-voice
```

This is the fastest path for a stable classroom demonstration.
