"""
visualize_results.py - Visualize collected gesture data and model accuracy

Creates charts to demonstrate dataset size/distribution and model performance.
Outputs PNG files into a local 'visualizations' folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "gesture" not in df.columns:
        raise ValueError("CSV must contain a 'gesture' column.")
    return df


def prepare_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c != "gesture"]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["gesture"].values)

    return X, y, label_encoder, feature_cols


def ensure_output_dir() -> Path:
    out_dir = Path("visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_class_distribution(df: pd.DataFrame, out_dir: Path):
    plt.figure(figsize=(7, 4))
    order = df["gesture"].value_counts().index
    sns.countplot(data=df, x="gesture", order=order, palette="viridis")
    plt.title("Collected Gesture Samples (Count per Class)")
    plt.xlabel("Gesture")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=130)
    plt.close()


def plot_pca_scatter(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder, out_dir: Path):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    for class_id, class_name in enumerate(label_encoder.classes_):
        idx = y == class_id
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], s=12, alpha=0.7, label=class_name)

    plt.title("Gesture Data (PCA 2D Projection)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scatter.png", dpi=130)
    plt.close()


def plot_accuracy(train_acc: float, test_acc: float, out_dir: Path):
    plt.figure(figsize=(4.5, 4))
    sns.barplot(x=["Train", "Test"], y=[train_acc, test_acc], palette="mako")
    plt.ylim(0, 1)
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    for i, v in enumerate([train_acc, test_acc]):
        plt.text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=130)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], out_dir: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=130)
    plt.close()


def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    cm = confusion_matrix(y_test, test_pred)

    return train_acc, test_acc, cm, classification_report(y_test, test_pred, output_dict=False)


def main():
    csv_path = "gesture_data.csv"
    out_dir = ensure_output_dir()

    df = load_data(csv_path)
    X, y, label_encoder, _ = prepare_features(df)

    # Visualize dataset
    plot_class_distribution(df, out_dir)
    plot_pca_scatter(X, y, label_encoder, out_dir)

    # Train + evaluate
    train_acc, test_acc, cm, report = train_and_evaluate(X, y)

    # Visualize metrics
    plot_accuracy(train_acc, test_acc, out_dir)
    plot_confusion_matrix(cm, list(label_encoder.classes_), out_dir)

    # Print summary for quick reporting
    print("===== DATASET SUMMARY =====")
    print(f"Total samples: {len(df)}")
    print("Samples per class:")
    print(df["gesture"].value_counts())

    print("\n===== MODEL ACCURACY =====")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(report)

    print(f"\nSaved visualizations to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
