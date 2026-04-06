# Project Structure

## Core runtime

- `main.py`: application entry point and thread orchestration
- `gesture_controller.py`: gesture detection and gesture-to-action logic
- `voice_assistant.py`: voice recognition and command handling
- `actions.py`: action model and queue transport
- `config.py`: centralized constants/configuration

## ML pipeline

- `collect_data.py`: collect labeled hand landmark data
- `train_model.py`: train and evaluate classifier (RF/SVM)
- `infer_live.py`: real-time model inference via webcam

## Support files

- `setup_check.py`: dependency/device checks
- `visualize_results.py`: visualization helper
- `generate_analysis_report.py`: report generation helper
- `requirements.txt`: Python dependencies

## Documentation

- `00_READ_ME_FIRST.md`: faculty-first guide
- `START_HERE.md`: quick orientation
- `QUICKSTART.md`: shortest command path
- `README.md`: complete usage guide
- `PROJECT_REPORT.md`: project report
- `DEBUGGING.md`: troubleshooting
- `SUBMISSION_CHECKLIST.md`: pre-upload checklist
