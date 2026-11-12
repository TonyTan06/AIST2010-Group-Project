import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 选择一段 Alternate Picking 音频
audio_path = "data/wav/alternate picking/class_0_carvin_dc-modern_1.wav"

# 读取音频
audio, sr = librosa.load(audio_path, sr=22050)

# 查看音频信息
print("Sample Rate:", sr)
print("Samples:", len(audio))
print("Duration:", len(audio) / sr, "seconds")

# Generate Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=128
)

# Convert power to decibels
mel_db = librosa.power_to_db(
    mel_spec,
    ref=np.max
)

print("Mel Spectrogram Shape:", mel_db.shape)

# Display Mel Spectrogram
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel",
    cmap="magma"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Alternate Picking - Mel Spectrogram")
plt.tight_layout()
plt.show()