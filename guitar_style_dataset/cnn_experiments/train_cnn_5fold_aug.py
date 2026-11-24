
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

from model_v4 import GuitarCNN
from augmentation import augment_logmel

import random

# ==================================================
# 1. Configuration
# ==================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 9
SEED = 42

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

output_dir = Path("results/cnn_v4_5fold_aug")
output_dir.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. Load dataset
# ==================================================

with np.load("cnn_experiments/cnn_dataset.npz") as data:

    X = data["X"]
    y = data["y"]
    source_files = data["source_files"]
    class_names = data["class_names"].tolist()

with open("data/folds.json", "r") as f:
    folds = json.load(f)

wav_names = np.array([
    Path(path).name
    for path in source_files
])

print("Dataset shape:", X.shape)
print("Number of folds:", len(folds))


# ==================================================
# 3. Extract exercise groups
# ==================================================

def get_exercise_group(filename):

    parts = Path(filename).stem.split("_", 4)

    assert len(parts) == 5, filename
    assert parts[0] == "class", filename

    return f"class_{parts[1]}_{parts[4]}"


groups = np.array([
    get_exercise_group(name)
    for name in wav_names
])

print("Total exercise groups:", len(np.unique(groups)))


# ==================================================
# 4. Create fold-specific data split
# ==================================================

def create_fold_split(fold_name):

    fold = folds[fold_name]

    original_train_indices = np.where(
        np.isin(wav_names, fold["train"])
    )[0]

    test_indices = np.where(
        np.isin(wav_names, fold["test"])
    )[0]

    # Select one validation exercise group per class
    rng = np.random.default_rng(SEED)

    validation_groups = []

    for class_id in range(NUM_CLASSES):

        class_indices = original_train_indices[
            y[original_train_indices] == class_id
        ]

        class_groups = np.unique(
            groups[class_indices]
        )

        assert len(class_groups) >= 2, (
            f"Not enough groups: "
            f"{fold_name}, class {class_id}"
        )

        selected_group = rng.choice(
            class_groups
        )

        validation_groups.append(
            selected_group
        )

    validation_mask = np.isin(
        groups[original_train_indices],
        validation_groups
    )

    validation_indices = original_train_indices[
        validation_mask
    ]

    train_indices = original_train_indices[
        ~validation_mask
    ]

    # Exercise-group leakage checks
    train_groups = set(groups[train_indices])
    val_groups = set(groups[validation_indices])
    test_groups = set(groups[test_indices])

    assert not (train_groups & val_groups), (
        f"{fold_name}: Train/Validation overlap"
    )

    assert not (train_groups & test_groups), (
        f"{fold_name}: Train/Test overlap"
    )

    assert not (val_groups & test_groups), (
        f"{fold_name}: Validation/Test overlap"
    )

    # Ensure all segments are included exactly once
    combined = np.concatenate([
        train_indices,
        validation_indices,
        test_indices
    ])

    assert len(combined) == len(y)
    assert len(np.unique(combined)) == len(y)

    # Ensure all nine classes exist in every split
    for indices in [
        train_indices,
        validation_indices,
        test_indices
    ]:

        assert set(y[indices]) == set(
            range(NUM_CLASSES)
        )

    return (
        train_indices,
        validation_indices,
        test_indices
    )


# ==================================================
# 5. Check all folds before training
# ==================================================

fold_splits = {}

print("\n===== Checking All Fold Splits =====")

for fold_name in sorted(folds.keys()):

    train_indices, val_indices, test_indices = (
        create_fold_split(fold_name)
    )

    fold_splits[fold_name] = (
        train_indices,
        val_indices,
        test_indices
    )

    print(
        f"{fold_name} | "
        f"Train: {len(train_indices)} | "
        f"Val: {len(val_indices)} | "
        f"Test: {len(test_indices)}"
    )

print("All fold leakage checks passed!")


# ==================================================
# 6. Create DataLoader
# ==================================================

def create_loader(indices, shuffle=False):

    features = torch.from_numpy(
        X[indices]
    ).float()

    labels = torch.from_numpy(
        y[indices]
    ).long()

    dataset = TensorDataset(
        features,
        labels
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0
    )


# ==================================================
# 7. Evaluation function
# ==================================================

def evaluate(model, loader, indices):

    model.eval()

    probabilities_list = []

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            probabilities_list.append(
                probabilities.cpu().numpy()
            )

    probabilities = np.concatenate(
        probabilities_list,
        axis=0
    )

    true_labels = y[indices]

    # Segment-level evaluation
    segment_predictions = probabilities.argmax(
        axis=1
    )

    segment_accuracy = accuracy_score(
        true_labels,
        segment_predictions
    )

    segment_f1 = f1_score(
        true_labels,
        segment_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    # Recording-level evaluation
    recording_sources = source_files[indices]

    recording_labels = []
    recording_predictions = []

    for recording in np.unique(recording_sources):

        mask = recording_sources == recording

        mean_probability = probabilities[
            mask
        ].mean(axis=0)

        predicted_class = mean_probability.argmax()

        true_classes = np.unique(
            true_labels[mask]
        )

        assert len(true_classes) == 1

        recording_labels.append(
            true_classes[0]
        )

        recording_predictions.append(
            predicted_class
        )

    recording_accuracy = accuracy_score(
        recording_labels,
        recording_predictions
    )

    recording_f1 = f1_score(
        recording_labels,
        recording_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        recording_labels,
        recording_predictions,
        labels=list(range(NUM_CLASSES))
    )

    return {
        "segment_accuracy": float(segment_accuracy),
        "segment_f1": float(segment_f1),
        "recording_accuracy": float(recording_accuracy),
        "recording_f1": float(recording_f1),
        "num_recordings": len(recording_labels),
        "confusion_matrix": matrix
    }


# ==================================================
# 8. Train one fold
# ==================================================

def train_fold(fold_name, split):

    print(f"\n{'=' * 70}")
    print(f"Training {fold_name}")
    print("=" * 70)

    train_indices, val_indices, test_indices = split

    # Reset random seeds for each independent model
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_loader = create_loader(
        train_indices,
        shuffle=True
    )

    val_loader = create_loader(
        val_indices,
        shuffle=False
    )

    test_loader = create_loader(
        test_indices,
        shuffle=False
    )

    print("Train segments:", len(train_indices))
    print("Validation segments:", len(val_indices))
    print("Test segments:", len(test_indices))

    # ----------------------------------------------
    # Calculate training class weights
    # ----------------------------------------------

    class_counts = np.bincount(
        y[train_indices],
        minlength=NUM_CLASSES
    )

    assert np.all(class_counts > 0)

    class_weights = (
        len(train_indices)
        / (NUM_CLASSES * class_counts)
    )

    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    # ----------------------------------------------
    # Create a new model for this fold
    # ----------------------------------------------

    model = GuitarCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_f1 = -1.0
    best_val_accuracy = 0.0
    best_epoch = 0

    model_path = (
        output_dir / f"{fold_name}_best.pt"
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_recording_accuracy": [],
        "val_macro_f1": []
    }

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        total_weight = 0.0

        correct = 0
        total = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Training-only Log-Mel augmentation.
            # Validation and test data remain untouched.
            inputs = augment_logmel(inputs)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            # Weighted epoch loss
            batch_weight = (
                weight_tensor[labels].sum().item()
            )

            running_loss += (
                loss.item() * batch_weight
            )

            total_weight += batch_weight

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        epoch_loss = (
            running_loss / total_weight
        )

        train_accuracy = correct / total

        # Validation
        val_metrics = evaluate(
            model,
            val_loader,
            val_indices
        )

        val_accuracy = val_metrics[
            "recording_accuracy"
        ]

        val_f1 = val_metrics[
            "recording_f1"
        ]

        print(
            f"{fold_name} | "
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"Val F1: {val_f1 * 100:.2f}%"
        )

        history["train_loss"].append(
            epoch_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_recording_accuracy"].append(
            val_accuracy
        )

        history["val_macro_f1"].append(
            val_f1
        )

        # Save best validation model
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            checkpoint = {
                "model_state_dict": {
                    name: value.cpu()
                    for name, value
                    in model.state_dict().items()
                },
                "class_names": class_names,
                "fold": fold_name,
                "epochs": best_epoch,
                "architecture": "CNN_V4",
                "validation_f1": best_val_f1,
                "validation_accuracy": best_val_accuracy,
                "class_weights": class_weights.tolist(),
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "seed": SEED
            }

            torch.save(
                checkpoint,
                model_path
            )

            print("  New best model saved!")

    # ----------------------------------------------
    # Save training history
    # ----------------------------------------------

    np.savez(
        output_dir / f"{fold_name}_history.npz",
        **{
            key: np.array(value)
            for key, value in history.items()
        }
    )

    # ----------------------------------------------
    # Load best model before testing
    # ----------------------------------------------

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    # Test set evaluated only after model selection
    test_metrics = evaluate(
        model,
        test_loader,
        test_indices
    )

    print(f"\n===== {fold_name} Test Results =====")

    print("Best epoch:", best_epoch)

    print(
        f"Best Validation F1: "
        f"{best_val_f1 * 100:.2f}%"
    )

    print(
        f"Test Segment Accuracy: "
        f"{test_metrics['segment_accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Recording Accuracy: "
        f"{test_metrics['recording_accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Recording Macro F1: "
        f"{test_metrics['recording_f1'] * 100:.2f}%"
    )

    print("Confusion Matrix:")

    print(test_metrics["confusion_matrix"])

    np.save(
        output_dir / f"{fold_name}_confusion.npy",
        test_metrics["confusion_matrix"]
    )

    return {
        "fold": fold_name,
        "best_epoch": best_epoch,
        "validation_accuracy": best_val_accuracy,
        "validation_f1": best_val_f1,
        "test_segment_accuracy": (
            test_metrics["segment_accuracy"]
        ),
        "test_segment_f1": (
            test_metrics["segment_f1"]
        ),
        "test_recording_accuracy": (
            test_metrics["recording_accuracy"]
        ),
        "test_recording_f1": (
            test_metrics["recording_f1"]
        ),
        "test_recordings": (
            test_metrics["num_recordings"]
        )
    }


# ==================================================
# 9. Run all five folds
# ==================================================

all_results = []

for fold_name, split in fold_splits.items():

    result = train_fold(
        fold_name,
        split
    )

    all_results.append(result)

    # Save completed fold results immediately
    with open(
        output_dir / "fold_results.csv",
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(result.keys())
        )

        writer.writeheader()
        writer.writerows(all_results)


# ==================================================
# 10. Calculate five-fold statistics
# ==================================================

recording_accuracies = np.array([
    result["test_recording_accuracy"]
    for result in all_results
])

recording_f1_scores = np.array([
    result["test_recording_f1"]
    for result in all_results
])

segment_accuracies = np.array([
    result["test_segment_accuracy"]
    for result in all_results
])

segment_f1_scores = np.array([
    result["test_segment_f1"]
    for result in all_results
])


# ==================================================
# 11. Final summary
# ==================================================

print("\n" + "=" * 70)
print("FINAL FIVE-FOLD RESULTS")
print("=" * 70)

for result in all_results:

    print(
        f"{result['fold']} | "
        f"Best Epoch: {result['best_epoch']:2d} | "
        f"Test Accuracy: "
        f"{result['test_recording_accuracy'] * 100:.2f}% | "
        f"Macro F1: "
        f"{result['test_recording_f1'] * 100:.2f}%"
    )

print("\nRecording-level metrics:")

print(
    f"Mean Accuracy: "
    f"{recording_accuracies.mean() * 100:.2f}%"
)

print(
    f"Accuracy Std: "
    f"{recording_accuracies.std() * 100:.2f}%"
)

print(
    f"Mean Macro F1: "
    f"{recording_f1_scores.mean() * 100:.2f}%"
)

print(
    f"Macro F1 Std: "
    f"{recording_f1_scores.std() * 100:.2f}%"
)

print("\nSegment-level metrics:")

print(
    f"Mean Accuracy: "
    f"{segment_accuracies.mean() * 100:.2f}%"
)

print(
    f"Mean Macro F1: "
    f"{segment_f1_scores.mean() * 100:.2f}%"
)


# ==================================================
# 12. Aggregate confusion matrix
# ==================================================

total_confusion = np.zeros(
    (NUM_CLASSES, NUM_CLASSES),
    dtype=np.int64
)

for fold_name in fold_splits:

    matrix = np.load(
        output_dir / f"{fold_name}_confusion.npy"
    )

    total_confusion += matrix

np.save(
    output_dir / "total_confusion_matrix.npy",
    total_confusion
)

print("\n===== Aggregate Confusion Matrix =====")

print(total_confusion)


# ==================================================
# 13. Save final summary
# ==================================================

summary = {
    "model": "CNN V4",
    "number_of_folds": len(all_results),
    "class_names": class_names,
    "recording_accuracy_mean": float(
        recording_accuracies.mean()
    ),
    "recording_accuracy_std": float(
        recording_accuracies.std()
    ),
    "recording_f1_mean": float(
        recording_f1_scores.mean()
    ),
    "recording_f1_std": float(
        recording_f1_scores.std()
    ),
    "segment_accuracy_mean": float(
        segment_accuracies.mean()
    ),
    "segment_f1_mean": float(
        segment_f1_scores.mean()
    ),
    "fold_results": all_results
}

with open(
    output_dir / "summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

print("\nAll experiments completed!")
print("Results saved to:", output_dir)
