import json
from pathlib import Path

import numpy as np


with np.load("cnn_experiments/cnn_dataset.npz") as data:
    y = data["y"]
    source_files = data["source_files"]
    class_names = data["class_names"]


with open("data/folds.json", "r") as f:
    folds = json.load(f)

fold = folds["fold_0"]

wav_names = np.array([
    Path(path).name
    for path in source_files
])

original_train_indices = np.where(
    np.isin(wav_names, fold["train"])
)[0]

test_indices = np.where(
    np.isin(wav_names, fold["test"])
)[0]

print("===== Original Training Distribution =====")

for idx, name in enumerate(class_names):

    class_indices = original_train_indices[
        y[original_train_indices] == idx
    ]

    num_segments = len(class_indices)

    num_recordings = len(
        np.unique(source_files[class_indices])
    )

    percentage = (
        num_segments / len(original_train_indices)
    ) * 100

    print(
        f"{name:20s} | "
        f"Recordings: {num_recordings:3d} | "
        f"Segments: {num_segments:4d} | "
        f"Percentage: {percentage:.2f}%"
    )

def get_exercise_group(filename):

    stem = Path(filename).stem

    parts = stem.split("_", 4)

    assert len(parts) == 5, filename
    assert parts[0] == "class", filename

    class_id = parts[1]
    exercise_id = parts[4]

    return f"class_{class_id}_{exercise_id}"


groups = np.array([
    get_exercise_group(name)
    for name in wav_names
])

print("\nExample WAV:", wav_names[0])
print("Example group:", groups[0])

print(
    "Total exercise groups:",
    len(np.unique(groups))
)

rng = np.random.default_rng(42)

validation_groups = []

for idx, name in enumerate(class_names):

    class_indices = original_train_indices[
        y[original_train_indices] == idx
    ]

    class_groups = np.unique(
        groups[class_indices]
    )

    assert len(class_groups) >= 2

    selected_group = rng.choice(
        class_groups
    )

    validation_groups.append(selected_group)

    print(
        f"{name:20s} | "
        f"Validation group: {selected_group}"
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

train_groups = set(groups[train_indices])

val_groups = set(groups[validation_indices])

test_groups = set(groups[test_indices])


assert not (train_groups & val_groups)

assert not (train_groups & test_groups)

assert not (val_groups & test_groups)


print("\n===== New Dataset Split =====")

print(
    "Train segments:",
    len(train_indices)
)

print(
    "Validation segments:",
    len(validation_indices)
)

print(
    "Test segments:",
    len(test_indices)
)

print(
    "Train recordings:",
    len(np.unique(source_files[train_indices]))
)

print(
    "Validation recordings:",
    len(np.unique(source_files[validation_indices]))
)

print(
    "Test recordings:",
    len(np.unique(source_files[test_indices]))
)

print("Group leakage check passed!")

output_path = Path(
    "cnn_experiments/fold0_split.npz"
)

np.savez(
    output_path,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices
)

print("Split saved to:", output_path)