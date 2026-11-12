import json
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import GuitarCNN


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# Load trained model
checkpoint = torch.load(
    "results/cnn_fold0_smoke.pt",
    map_location="cpu"
)

class_names = checkpoint["class_names"]

model = GuitarCNN(
    num_classes=len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

print("Model loaded successfully")

# Load dataset
with np.load("cnn_experiments/cnn_dataset.npz") as data:
    X = data["X"]
    y = data["y"]
    source_files = data["source_files"]

    assert data["class_names"].tolist() == class_names


# Load official folds
with open("data/folds.json", "r") as f:
    folds = json.load(f)

fold = folds["fold_0"]

wav_names = np.array([
    Path(path).name
    for path in source_files
])

test_indices = np.where(
    np.isin(wav_names, fold["test"])
)[0]


# Create test dataset
X_test = torch.from_numpy(
    X[test_indices]
).float()

y_test = torch.from_numpy(
    y[test_indices]
).long()

test_sources = source_files[test_indices]


test_dataset = TensorDataset(
    X_test,
    y_test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test segments:", len(test_indices))
print("Test recordings:", len(np.unique(test_sources)))

all_probabilities = []
all_labels = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )


all_probabilities = np.concatenate(
    all_probabilities,
    axis=0
)

all_labels = np.concatenate(
    all_labels,
    axis=0
)

predictions = np.argmax(
    all_probabilities,
    axis=1
)

print("Prediction shape:", all_probabilities.shape)

class_ids = list(range(len(class_names)))

segment_accuracy = accuracy_score(
    all_labels,
    predictions
)

segment_f1 = f1_score(
    all_labels,
    predictions,
    labels=class_ids,
    average="macro",
    zero_division=0
)

print("\n===== Segment-level Results =====")

print(
    f"Accuracy: {segment_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {segment_f1 * 100:.2f}%"
)

recording_predictions = []
recording_labels = []

unique_recordings = np.unique(
    test_sources
)

for recording in unique_recordings:

    # Find all segments from this recording
    mask = test_sources == recording

    # Average their probabilities
    mean_probability = all_probabilities[
        mask
    ].mean(axis=0)

    # Recording-level prediction
    predicted_class = np.argmax(
        mean_probability
    )

    true_classes = np.unique(
        all_labels[mask]
    )

    assert len(true_classes) == 1

    true_class = true_classes[0]

    recording_predictions.append(
        predicted_class
    )

    recording_labels.append(
        true_class
    )


recording_accuracy = accuracy_score(
    recording_labels,
    recording_predictions
)

recording_f1 = f1_score(
    recording_labels,
    recording_predictions,
    labels=class_ids,
    average="macro",
    zero_division=0
)


print("\n===== Recording-level Results =====")

print(
    "Total recordings:",
    len(recording_labels)
)

print(
    f"Accuracy: {recording_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: {recording_f1 * 100:.2f}%"
)

print("\n===== Classification Report =====")

print(
    classification_report(
        recording_labels,
        recording_predictions,
        labels=class_ids,
        target_names=class_names,
        zero_division=0
    )
)

print("\n===== Confusion Matrix =====")

print(
    confusion_matrix(
        recording_labels,
        recording_predictions,
        labels=class_ids
    )
)