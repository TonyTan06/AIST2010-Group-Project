
import argparse
from pathlib import Path

import librosa
import numpy as np
import torch

from model_v4 import GuitarCNN


# ==================================================
# 1. Configuration
# ==================================================

SAMPLE_RATE = 22050
SEGMENT_LENGTH = SAMPLE_RATE

N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128

MODEL_PATH = "results/cnn_v4_5fold/fold_0_best.pt"

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ==================================================
# 2. Load trained CNN
# ==================================================

def load_model(model_path):

    checkpoint = torch.load(
        model_path,
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

    return model, class_names


# ==================================================
# 3. Audio preprocessing
# ==================================================

def preprocess_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    duration = len(audio) / sr

    print(f"Audio duration: {duration:.2f} seconds")

    if len(audio) < SEGMENT_LENGTH:

        raise ValueError(
            "Audio must be at least 1 second long."
        )

    mel_spectrograms = []

    for start in range(
        0,
        len(audio),
        SEGMENT_LENGTH
    ):

        end = start + SEGMENT_LENGTH

        segment = audio[start:end]

        # Keep complete one-second segments
        if len(segment) != SEGMENT_LENGTH:
            continue

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

        mel_spectrograms.append(
            mel_db.astype(np.float32)
        )

    X = np.array(
        mel_spectrograms,
        dtype=np.float32
    )

    # Add channel dimension
    X = np.expand_dims(
        X,
        axis=1
    )

    print("Number of segments:", len(X))
    print("CNN input shape:", X.shape)

    return torch.from_numpy(X)


# ==================================================
# 4. Predict playing technique
# ==================================================

def predict_audio(model, class_names, audio_path):

    inputs = preprocess_audio(
        audio_path
    )

    all_probabilities = []

    with torch.no_grad():

        # Predict in small batches
        for batch in inputs.split(32):

            batch = batch.to(device)

            outputs = model(batch)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    # Recording-level prediction
    mean_probabilities = probabilities.mean(
        axis=0
    )

    predicted_class = int(
        np.argmax(mean_probabilities)
    )

    technique = class_names[
        predicted_class
    ]

    confidence = float(
        mean_probabilities[predicted_class]
    )

    return (
        technique,
        confidence,
        mean_probabilities
    )


# ==================================================
# 5. Main program
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Electric Guitar Technique Classifier"
    )

    parser.add_argument(
        "audio",
        help="Path to the input audio file"
    )

    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to a trained CNN V4 model"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.is_file():

        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    print("Using device:", device)
    print("Loading model:", args.model)

    model, class_names = load_model(
        args.model
    )

    print("Model loaded successfully")

    print("\n===== Audio Information =====")
    print("File:", audio_path)

    (
        technique,
        confidence,
        probabilities
    ) = predict_audio(
        model,
        class_names,
        audio_path
    )

    print("\n===== Prediction Result =====")

    print(
        "Predicted technique:",
        technique
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("\n===== All Class Probabilities =====")

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    for idx in sorted_indices:

        print(
            f"{class_names[idx]:20s} | "
            f"{probabilities[idx] * 100:6.2f}%"
        )


if __name__ == "__main__":

    main()
