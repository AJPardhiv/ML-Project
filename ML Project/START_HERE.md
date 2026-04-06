# Start Here

This guide provides the fastest path for evaluation.

## Step 1: Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python setup_check.py
```

## Step 2: Run core demo

```bash
python main.py --no-voice
```

Show these actions during demo:
- move cursor
- click gesture
- scroll gesture
- pause gesture

## Step 3: Optional voice demo

```bash
python main.py
```

Voice mode requires a valid Vosk model in `model/`.

## Step 4: Optional ML demo

```bash
python collect_data.py --output gesture_data.csv --samples 50
python train_model.py --data gesture_data.csv --type rf --output models/gesture_model.joblib
python infer_live.py --model models/gesture_model.joblib --threshold 0.6
```

## Files to review

- `README.md`
- `QUICKSTART.md`
- `PROJECT_REPORT.md`
- `DEBUGGING.md`
- `SUBMISSION_CHECKLIST.md`
