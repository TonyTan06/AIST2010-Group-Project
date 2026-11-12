
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from model_v4 import GuitarCNN


# ==================================================
# 1. Training configuration
# ==================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30
NUM_CLASSES = 9

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# ==================================================
# 2. Load CNN dataset
# ==================================================

with np.load("cnn_experiments/cnn_dataset.npz") as data:

    X = data["X"]
    y = data["y"]
    source_files = data["source_files"]
    class_names = data["class_names"]

print("Dataset shape:", X.shape)


# ==================================================
# 3. Load Train / Validation split
# ==================================================

with np.load(
    "cnn_experiments/fold0_split.npz"
) as split:

    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]

print("Train segments:", len(train_indices))
print("Validation segments:", len(validation_indices))

assert len(train_indices) == 4117
assert len(validation_indices) == 887

assert not (
    set(source_files[train_indices])
    & set(source_files[validation_indices])
), "Data leakage detected!"


# ==================================================
# 4. Create training DataLoader
# ==================================================

X_train = torch.from_numpy(
    X[train_indices]
).float()

y_train = torch.from_numpy(
    y[train_indices]
).long()

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training batches:", len(train_loader))


# ==================================================
# 5. Create validation DataLoader
# ==================================================

X_val = torch.from_numpy(
    X[validation_indices]
).float()

y_val = torch.from_numpy(
    y[validation_indices]
).long()

val_sources = source_files[validation_indices]
val_labels = y[validation_indices]

val_dataset = TensorDataset(
    X_val,
    y_val
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation batches:", len(val_loader))


# ==================================================
# 6. Calculate class weights
# ==================================================

# Count segments in each training class
class_counts = np.bincount(
    y[train_indices],
    minlength=NUM_CLASSES
)

# Make sure every class exists in training data
assert np.all(class_counts > 0)

# Balanced class weights
class_weights = (
    len(train_indices)
    / (NUM_CLASSES * class_counts)
)

print("\n===== Training Class Weights =====")

for idx, name in enumerate(class_names):

    print(
        f"{name:20s} | "
        f"Segments: {class_counts[idx]:4d} | "
        f"Weight: {class_weights[idx]:.4f}"
    )

# Convert weights to GPU tensor
weight_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)


# ==================================================
# 7. Initialize CNN model
# ==================================================

model = GuitarCNN(
    num_classes=NUM_CLASSES
).to(device)

# Weighted CrossEntropyLoss
criterion = nn.CrossEntropyLoss(
    weight=weight_tensor
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# 8. Validation function
# ==================================================

def evaluate_validation(model):

    model.eval()

    all_probabilities = []

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    all_probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    # Segment-level evaluation
    segment_predictions = np.argmax(
        all_probabilities,
        axis=1
    )

    segment_accuracy = accuracy_score(
        val_labels,
        segment_predictions
    )

    # Recording-level evaluation
    recording_predictions = []
    recording_labels = []

    for recording in np.unique(val_sources):

        mask = val_sources == recording

        mean_probability = all_probabilities[
            mask
        ].mean(axis=0)

        predicted_class = np.argmax(
            mean_probability
        )

        true_classes = np.unique(
            val_labels[mask]
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
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    return (
        segment_accuracy,
        recording_accuracy,
        recording_f1
    )


# ==================================================
# 9. Initialize best model tracking
# ==================================================

best_val_f1 = -1.0
best_val_accuracy = 0.0
best_epoch = 0

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

model_path = (
    output_dir / "cnn_fold0_v4_best.pt"
)

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_segment_accuracy": [],
    "val_recording_accuracy": [],
    "val_macro_f1": []
}


# ==================================================
# 10. Training loop
# ==================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    total_weight = 0.0

    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for inputs, labels in progress:

        # Move current batch to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate weighted loss
        loss = criterion(
            outputs,
            labels
        )

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Calculate the total weight of this batch
        batch_weight = (
            weight_tensor[labels].sum().item()
        )

        # Record weighted training loss
        running_loss += (
            loss.item() * batch_weight
        )

        total_weight += batch_weight

        # Record training accuracy
        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    # Calculate epoch statistics
    epoch_loss = running_loss / total_weight

    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: "
        f"{epoch_accuracy * 100:.2f}%"
    )

    # ----------------------------------------------
    # Evaluate validation set
    # ----------------------------------------------

    (
        val_segment_acc,
        val_recording_acc,
        val_f1
    ) = evaluate_validation(model)

    print(
        f"Validation | "
        f"Segment Accuracy: "
        f"{val_segment_acc * 100:.2f}% | "
        f"Recording Accuracy: "
        f"{val_recording_acc * 100:.2f}% | "
        f"Macro F1: {val_f1 * 100:.2f}%"
    )

    # ----------------------------------------------
    # Record training history
    # ----------------------------------------------

    history["train_loss"].append(
        epoch_loss
    )

    history["train_accuracy"].append(
        epoch_accuracy
    )

    history["val_segment_accuracy"].append(
        val_segment_acc
    )

    history["val_recording_accuracy"].append(
        val_recording_acc
    )

    history["val_macro_f1"].append(
        val_f1
    )

    # ----------------------------------------------
    # Save best model
    # ----------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_val_accuracy = val_recording_acc
        best_epoch = epoch + 1

        checkpoint = {

            "model_state_dict": {
                name: value.cpu()
                for name, value
                in model.state_dict().items()
            },

            "class_names": class_names.tolist(),

            "fold": "fold_0",
            "architecture": "CNN_V4",

            "epochs": best_epoch,

            "validation_f1": float(
                best_val_f1
            ),

            "validation_accuracy": float(
                best_val_accuracy
            ),

            "batch_size": BATCH_SIZE,

            "learning_rate": LEARNING_RATE,

            "loss_function": (
                "WeightedCrossEntropyLoss"
            ),

            "class_weights": (
                class_weights.tolist()
            ),

            "class_counts": (
                class_counts.tolist()
            )
        }

        torch.save(
            checkpoint,
            model_path
        )

        print(
            f"New best model saved! "
            f"Epoch: {best_epoch} | "
            f"F1: {best_val_f1 * 100:.2f}%"
        )

    print("-" * 70)


# ==================================================
# 11. Save training history
# ==================================================

history_path = (
    output_dir / "cnn_fold0_v4_history.npz"
)

np.savez(
    history_path,

    train_loss=np.array(
        history["train_loss"]
    ),

    train_accuracy=np.array(
        history["train_accuracy"]
    ),

    val_segment_accuracy=np.array(
        history["val_segment_accuracy"]
    ),

    val_recording_accuracy=np.array(
        history["val_recording_accuracy"]
    ),

    val_macro_f1=np.array(
        history["val_macro_f1"]
    )
)


# ==================================================
# 12. Final training summary
# ==================================================

print("\n===== Training Completed =====")

print("Model: CNN V4")

print("Loss Function: Weighted CrossEntropyLoss")

print("Total epochs:", EPOCHS)

print("Best epoch:", best_epoch)

print(
    f"Best validation Recording Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Best validation Macro F1: "
    f"{best_val_f1 * 100:.2f}%"
)

print("Best model saved to:", model_path)

print("Training history saved to:", history_path)
