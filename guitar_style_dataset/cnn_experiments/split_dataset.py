import json
from pathlib import Path

import numpy as np

DATASET_PATH = "cnn_experiments/cnn_dataset.npz"
FOLDS_PATH = "data/folds.json"

# Load dataset metadata
with np.load(DATASET_PATH) as dataset:
    y = dataset["y"]
    source_files = dataset["source_files"]
    class_names = dataset["class_names"]

# Load official folds
with open(FOLDS_PATH, "r") as f:
    folds = json.load(f)

print("Number of folds:", len(folds))
print("Total segments:", len(y))

wav_names = np.array([
    Path(path).name
    for path in source_files
])

print("Example source path:", source_files[0])
print("Example WAV name:", wav_names[0])

all_test_files = set()

for fold_name, fold_data in folds.items():

    train_files = fold_data["train"]
    test_files = fold_data["test"]

    # Find the corresponding segment indices
    train_indices = np.where(
        np.isin(wav_names, train_files)
    )[0]

    test_indices = np.where(
        np.isin(wav_names, test_files)
    )[0]

    # Check for data leakage
    train_sources = set(wav_names[train_indices])
    test_sources = set(wav_names[test_indices])

    overlap = train_sources & test_sources

    all_test_files.update(test_sources)

    print("\n==========", fold_name, "==========")

    print("Train recordings:", len(train_sources))
    print("Test recordings:", len(test_sources))

    print("Train segments:", len(train_indices))
    print("Test segments:", len(test_indices))

    print("Overlapping recordings:", len(overlap))

    assert len(train_indices) > 0
    assert len(test_indices) > 0

    assert len(overlap) == 0, "Data leakage detected!"

    all_recordings = set(wav_names)

untested_files = all_recordings - all_test_files

print("\n========== Summary ==========")

print("Total recordings:", len(all_recordings))

print("Recordings tested across all folds:",
      len(all_test_files))

print("Recordings never tested:",
      len(untested_files))

for filename in sorted(untested_files):
    print(filename)

print("\nFold checking completed.")