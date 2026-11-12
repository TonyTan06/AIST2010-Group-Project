import librosa
import numpy as np

def audio_to_mel(segment, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    return mel_db

audio_path = "data/wav/alternate picking/class_0_carvin_dc-modern_1.wav"

audio, sr = librosa.load(audio_path, sr=22050)

# 每个片段的时长
segment_duration = 1

# 每个片段的采样点数量
segment_length = int(sr * segment_duration)

print("Segment length:", segment_length)

segments = []

for start in range(0, len(audio), segment_length):
    end = start + segment_length

    segment = audio[start:end]

    # 只保留完整的 1 秒片段
    if len(segment) == segment_length:
        segments.append(segment)

print("Number of segments:", len(segments))

segments = np.array(segments)

print("Segments shape:", segments.shape)

mel_spectrograms = []

for segment in segments:
    mel_db = audio_to_mel(segment, sr)

    mel_spectrograms.append(mel_db)

mel_spectrograms = np.array(
    mel_spectrograms,
    dtype=np.float32
)

print("Mel spectrograms shape:", mel_spectrograms.shape)

cnn_input = np.expand_dims(
    mel_spectrograms,
    axis=1
)

print("CNN input shape:", cnn_input.shape)