# One-Page Slide Notes (Gesture Project)

## Slide Title
Gesture-Based Laptop Controller: One-Page Model Summary

## Dataset
- Total samples: 4600
- Features: 63 landmark coordinates (21 landmarks × x/y/z)
- Target classes: move, click, scroll, pause
- Missing values: 0

## Model Architecture
- Input: MediaPipe hand landmarks
- Preprocessing: NaN-to-zero, label encoding, stratified 80/20 split
- Classifier: RandomForest (n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2)

## Key Results
- Test Accuracy: 0.9957
- Weighted Precision: 0.9957
- Weighted Recall: 0.9957
- Weighted F1-score: 0.9957
- ROC-AUC (micro): 1.0000
- PR-AUC (micro): 0.9999

## Plot Captions
- OOB Error Curve: Error stabilizes quickly as trees increase, indicating robust ensemble behavior.
- ROC Curves: All classes show near-perfect separability.
- Precision-Recall Curves: Very high precision/recall trade-off across classes.
- Confusion Matrix: Minimal class confusion and consistently correct predictions.
