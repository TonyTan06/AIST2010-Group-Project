from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

DATA_DIR = Path("data/wav")

SAMPLE_RATE = 22050
SEGMENT_DURATION = 1
SEGMENT_LENGTH = SAMPLE_RATE * SEGMENT_DURATION

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

class_names = sorted(
    folder.name
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
)

class_to_idx = {
    name: idx
    for idx, name in enumerate(class_names)
}

print("Class mapping:")

for name, idx in class_to_idx.items():
    print(idx, name)

def audio_to_mel(segment):
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    return mel_db.astype(np.float32)

X = []
y = []
source_files = []
segment_indices = []

for class_name in class_names:

    class_dir = DATA_DIR / class_name
    wav_files = sorted(class_dir.glob("*.wav"))

    label = class_to_idx[class_name]

    for wav_path in tqdm(
        wav_files,
        desc=class_name
    ):

        audio, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE
        )

        for start in range(
            0,
            len(audio),
            SEGMENT_LENGTH
        ):

            end = start + SEGMENT_LENGTH

            segment = audio[start:end]

            if len(segment) != SEGMENT_LENGTH:
                continue

            mel_db = audio_to_mel(segment)

            X.append(mel_db)
            y.append(label)

            source_files.append(
                wav_path.relative_to(DATA_DIR).as_posix()
            )

            segment_indices.append(
                start // SEGMENT_LENGTH
            )

X = np.array(X, dtype=np.float32)
X = np.expand_dims(X, axis=1)

y = np.array(y, dtype=np.int64)

source_files = np.array(source_files)
segment_indices = np.array(
    segment_indices,
    dtype=np.int64
)

print("\nDataset information:")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Source files shape:", source_files.shape)
print("Segment indices shape:", segment_indices.shape)

print("Original recordings represented:",
      len(np.unique(source_files)))

print("Total segments:", len(X))

output_path = Path(
    "cnn_experiments/cnn_dataset.npz"
)

np.savez(
    output_path,
    X=X,
    y=y,
    source_files=source_files,
    segment_indices=segment_indices,
    class_names=np.array(class_names)
)

print("Dataset saved to:", output_path)