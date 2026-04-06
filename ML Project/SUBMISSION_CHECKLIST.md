# Submission Checklist

Use this checklist before uploading for faculty evaluation.

## 1. Functional Validation

- [ ] `python setup_check.py` completes without failure.
- [ ] `python main.py --no-voice` launches and gesture controls respond.
- [ ] `python main.py` runs with voice mode when `model/` is present.
- [ ] `python train_model.py --data gesture_data.csv --type rf --output models/gesture_model.joblib` runs (optional).
- [ ] `python infer_live.py --model models/gesture_model.joblib` runs (optional).

## 2. Documentation Validation

- [ ] `README.md` command examples are accurate.
- [ ] `QUICKSTART.md` is enough for first execution.
- [ ] `PROJECT_REPORT.md` is included in final upload.

## 3. Package Cleanliness

- [ ] Remove personal/local files not needed for evaluation.
- [ ] Ensure virtual environment folders are excluded.
- [ ] Ensure large temporary archives are excluded.
- [ ] Include source code, requirements, and required docs.

## 4. Recommended Upload Contents

- Python source files (`*.py`)
- `requirements.txt`
- `README.md`, `QUICKSTART.md`, `START_HERE.md`, `PROJECT_REPORT.md`
- `model/` (if voice demo is required)
- `models/gesture_model.joblib` (if ML demo is required)
