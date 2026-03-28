"""
generate_analysis_report.py

Generates a complete mini-report package for the gesture dataset:
1) Dataset description with attributes
2) Model/method architecture summary
3) Tentative results (metrics-heavy)
4) Visualization graphs (loss/error curve, ROC, PR, histograms, etc.)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize


RANDOM_STATE = 42
DATA_PATH = Path("gesture_data.csv")
OUT_DIR = Path("visualizations")
REPORT_PATH = Path("RESULTS_ANALYSIS.md")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "gesture" not in df.columns:
        raise ValueError("Expected target column 'gesture' in dataset")
    return df


def prepare_data(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col != "gesture"]
    X = np.nan_to_num(df[feature_cols].values, nan=0.0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["gesture"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X, y, X_train, X_test, y_train, y_test, feature_cols, encoder


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def ensure_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    order = df["gesture"].value_counts().index
    sns.countplot(data=df, x="gesture", order=order, palette="viridis")
    plt.title("Class Distribution of Gestures")
    plt.xlabel("Gesture")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
    plt.close()


def plot_xyz_histograms(df: pd.DataFrame) -> None:
    x_cols = [c for c in df.columns if c.endswith("_x")]
    y_cols = [c for c in df.columns if c.endswith("_y")]
    z_cols = [c for c in df.columns if c.endswith("_z")]

    x_vals = df[x_cols].values.flatten()
    y_vals = df[y_cols].values.flatten()
    z_vals = df[z_cols].values.flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(x_vals, bins=40, color="#4c78a8", alpha=0.8)
    axes[0].set_title("Histogram of X Coordinates")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(y_vals, bins=40, color="#59a14f", alpha=0.8)
    axes[1].set_title("Histogram of Y Coordinates")
    axes[1].set_xlabel("Y")

    axes[2].hist(z_vals, bins=40, color="#e15759", alpha=0.8)
    axes[2].set_title("Histogram of Z Coordinates")
    axes[2].set_xlabel("Z")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "xyz_histograms.png", dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_feature_importance(model: RandomForestClassifier, feature_cols: list[str], top_n: int = 15) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    top_features = [feature_cols[i] for i in idx]
    top_importances = importances[idx]

    plt.figure(figsize=(9, 6))
    plt.barh(top_features, top_importances, color="#72b7b2")
    plt.title(f"Top {top_n} Feature Importances (RandomForest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "feature_importance_top15.png", dpi=150)
    plt.close()


def plot_oob_error_curve(X_train: np.ndarray, y_train: np.ndarray) -> tuple[list[int], list[float]]:
    estimators_range = list(range(10, 210, 10))
    oob_errors = []

    for n_estimators in estimators_range:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            oob_score=True,
            bootstrap=True,
        )
        clf.fit(X_train, y_train)
        oob_errors.append(1 - clf.oob_score_)

    plt.figure(figsize=(8, 5))
    plt.plot(estimators_range, oob_errors, marker="o", color="#f28e2b")
    plt.title("Model Error Curve (OOB Error vs Number of Trees)")
    plt.xlabel("Number of Trees")
    plt.ylabel("OOB Error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "oob_error_curve.png", dpi=150)
    plt.close()

    return estimators_range, oob_errors


def plot_learning_curves(X: np.ndarray, y: np.ndarray) -> None:
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        X=X,
        y=y,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
    plt.plot(train_sizes, val_mean, marker="s", label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "learning_curve.png", dpi=150)
    plt.close()


def plot_roc_pr_curves(y_test: np.ndarray, y_proba: np.ndarray, class_names: np.ndarray) -> tuple[float, float]:
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))

    fpr = {}
    tpr = {}
    roc_auc = {}
    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Micro-average PR
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_proba.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{name} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2, label=f"micro-average (AUC={roc_auc['micro']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.title("Multi-class ROC Curves (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve_multiclass.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 6))
    for i, name in enumerate(class_names):
        plt.plot(recall[i], precision[i], label=f"{name} (AUC={pr_auc[i]:.3f})")
    plt.plot(
        recall["micro"],
        precision["micro"],
        linestyle="--",
        linewidth=2,
        label=f"micro-average (AUC={pr_auc['micro']:.3f})",
    )
    plt.title("Multi-class Precision-Recall Curves (One-vs-Rest)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "precision_recall_curve_multiclass.png", dpi=150)
    plt.close()

    return float(roc_auc["micro"]), float(pr_auc["micro"])


def build_metrics(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    roc_micro_auc: float,
    pr_micro_auc: float,
) -> dict:
    return {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_precision_weighted": float(precision_score(y_test, y_test_pred, average="weighted")),
        "test_recall_weighted": float(recall_score(y_test, y_test_pred, average="weighted")),
        "test_f1_weighted": float(f1_score(y_test, y_test_pred, average="weighted")),
        "roc_micro_auc": roc_micro_auc,
        "pr_micro_auc": pr_micro_auc,
    }


def dataset_description(df: pd.DataFrame) -> str:
    total_rows = len(df)
    total_cols = len(df.columns)
    feature_cols = [col for col in df.columns if col != "gesture"]
    missing_cells = int(df.isna().sum().sum())

    class_counts = df["gesture"].value_counts().sort_index()
    class_lines = "\n".join([f"- {name}: {count} samples" for name, count in class_counts.items()])

    attribute_lines = [
        "- `landmark_0_x` to `landmark_20_x`: X-coordinates of 21 hand landmarks (normalized)",
        "- `landmark_0_y` to `landmark_20_y`: Y-coordinates of 21 hand landmarks (normalized)",
        "- `landmark_0_z` to `landmark_20_z`: Z/depth values of 21 hand landmarks (normalized)",
        "- `gesture`: Target class label (`move`, `click`, `scroll`, `pause`)",
    ]

    return (
        f"### 1) Dataset Description (With Attributes)\n\n"
        f"- Total samples: **{total_rows}**\n"
        f"- Total attributes: **{total_cols}** (63 input features + 1 target label)\n"
        f"- Feature columns: **{len(feature_cols)}**\n"
        f"- Missing values: **{missing_cells}**\n"
        f"- Class distribution:\n{class_lines}\n\n"
        f"**Attributes**\n"
        + "\n".join(attribute_lines)
    )


def architecture_description() -> str:
    return (
        "### 2) Model/Method Architecture\n\n"
        "**Input Layer**\n"
        "- Per frame: 21 MediaPipe hand landmarks\n"
        "- Flattened features: 21 × 3 = 63 values\n\n"
        "**Data Pipeline**\n"
        "- CSV loading (`gesture_data.csv`)\n"
        "- NaN handling using zero imputation\n"
        "- Label encoding for gesture classes\n"
        "- Stratified train/test split (80/20)\n\n"
        "**Classifier**\n"
        "- RandomForestClassifier\n"
        "- Hyperparameters: `n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`\n\n"
        "**Inference Flow**\n"
        "1. Capture hand landmarks\n"
        "2. Form 63-D feature vector\n"
        "3. Predict class + probability distribution\n"
        "4. Apply action mapping (`move`, `click`, `scroll`, `pause`)\n"
    )


def results_description(metrics: dict, report_text: str) -> str:
    return (
        "### 3) Tentative Results (Results-Focused)\n\n"
        "This section intentionally emphasizes empirical outcomes from the provided sample dataset.\n\n"
        "**Overall Metrics**\n"
        f"- Training Accuracy: **{metrics['train_accuracy']:.4f}**\n"
        f"- Test Accuracy: **{metrics['test_accuracy']:.4f}**\n"
        f"- Weighted Precision: **{metrics['test_precision_weighted']:.4f}**\n"
        f"- Weighted Recall: **{metrics['test_recall_weighted']:.4f}**\n"
        f"- Weighted F1-score: **{metrics['test_f1_weighted']:.4f}**\n"
        f"- Micro-average ROC AUC: **{metrics['roc_micro_auc']:.4f}**\n"
        f"- Micro-average PR AUC: **{metrics['pr_micro_auc']:.4f}**\n\n"
        "**Per-Class Performance (classification report)**\n\n"
        "```\n"
        f"{report_text}\n"
        "```\n"
    )


def visualization_description() -> str:
    return (
        "### 4) Visualizations in Graphs\n\n"
        "Generated and saved under `visualizations/`:\n\n"
        "- `class_distribution.png` → class/sample count distribution\n"
        "- `xyz_histograms.png` → histograms of landmark X/Y/Z coordinate distributions\n"
        "- `confusion_matrix.png` → class-wise prediction performance\n"
        "- `feature_importance_top15.png` → most informative landmark features\n"
        "- `oob_error_curve.png` → model error curve (loss-like curve using OOB error)\n"
        "- `learning_curve.png` → train/validation learning behavior\n"
        "- `roc_curve_multiclass.png` → one-vs-rest ROC curves with AUC\n"
        "- `precision_recall_curve_multiclass.png` → one-vs-rest PR curves with AUC\n"
    )


def write_report(df: pd.DataFrame, metrics: dict, report_text: str) -> None:
    sections = [
        "# Gesture Project Analysis Report\n",
        dataset_description(df),
        architecture_description(),
        results_description(metrics, report_text),
        visualization_description(),
    ]
    REPORT_PATH.write_text("\n\n".join(sections), encoding="utf-8")


def main() -> None:
    ensure_output()
    df = load_data(DATA_PATH)

    X, y, X_train, X_test, y_train, y_test, feature_cols, encoder = prepare_data(df)

    model = train_model(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, target_names=encoder.classes_)

    plot_class_distribution(df)
    plot_xyz_histograms(df)
    plot_confusion(cm, encoder.classes_)
    plot_feature_importance(model, feature_cols)
    _, _ = plot_oob_error_curve(X_train, y_train)
    plot_learning_curves(X, y)
    roc_micro_auc, pr_micro_auc = plot_roc_pr_curves(y_test, y_proba, encoder.classes_)

    metrics = build_metrics(
        y_train=y_train,
        y_train_pred=y_train_pred,
        y_test=y_test,
        y_test_pred=y_test_pred,
        roc_micro_auc=roc_micro_auc,
        pr_micro_auc=pr_micro_auc,
    )

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(df, metrics, class_report)

    print("Report generated:", REPORT_PATH.resolve())
    print("Visualizations saved in:", OUT_DIR.resolve())
    print("Metrics summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
