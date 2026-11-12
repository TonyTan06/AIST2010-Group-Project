import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from model import GuitarCNN


# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 5
NUM_CLASSES = 9

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

# Load dataset
with np.load("cnn_experiments/cnn_dataset.npz") as data:
    X = data["X"]
    y = data["y"]
    source_files = data["source_files"]
    class_names = data["class_names"]

print("Dataset shape:", X.shape)

# Load official folds
with open("data/folds.json", "r") as f:
    folds = json.load(f)

# Select Fold 0
fold = folds["fold_0"]

wav_names = np.array([
    Path(path).name
    for path in source_files
])

train_indices = np.where(
    np.isin(wav_names, fold["train"])
)[0]

test_indices = np.where(
    np.isin(wav_names, fold["test"])
)[0]

print("Train segments:", len(train_indices))
print("Test segments:", len(test_indices))

assert len(train_indices) == 5004
assert len(test_indices) == 1040

assert not (
    set(wav_names[train_indices])
    & set(wav_names[test_indices])
), "Data leakage detected!"

# Convert training data into PyTorch tensors
X_train = torch.from_numpy(
    X[train_indices]
).float()

y_train = torch.from_numpy(
    y[train_indices]
).long()

# Create training dataset
train_dataset = TensorDataset(
    X_train,
    y_train
)

# Create DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training batches:", len(train_loader))

# Create CNN model
model = GuitarCNN(
    num_classes=NUM_CLASSES
).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# Training loop
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
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

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Record training statistics
        running_loss += (
            loss.item() * inputs.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total

    epoch_accuracy = correct / total * 100

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: {epoch_accuracy:.2f}%"
    )

# Save trained model
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

model_path = output_dir / "cnn_fold0_smoke.pt"

checkpoint = {
    "model_state_dict": {
        name: value.cpu()
        for name, value in model.state_dict().items()
    },
    "class_names": class_names.tolist(),
    "fold": "fold_0",
    "epochs": EPOCHS
}

torch.save(checkpoint, model_path)

print("Training completed!")
print("Model saved to:", model_path)