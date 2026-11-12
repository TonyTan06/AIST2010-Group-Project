from pathlib import Path
import tempfile
import io
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import torch

from predict_audio import (
    load_model,
    preprocess_audio
)

import altair as alt
import librosa
from PIL import Image, ImageDraw, ImageFont


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Guitar AI",
    page_icon="🎸",
    layout="wide"
)


# ==================================================
# 2. Model configuration
# ==================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "cnn_v4_5fold"
    / "fold_0_best.pt"
)


# ==================================================
# 3. Load model
# ==================================================

@st.cache_resource
def get_model():

    model, class_names = load_model(
        MODEL_PATH
    )

    return model, class_names


# ==================================================
# Spectrogram preprocessing
# ==================================================

@st.cache_data(max_entries=8)
def get_spectrograms(audio_bytes):

    with tempfile.TemporaryDirectory() as tmp_dir:

        audio_path = Path(tmp_dir) / "input.wav"

        audio_path.write_bytes(audio_bytes)

        inputs = preprocess_audio(audio_path)

    return inputs.detach().cpu().numpy()


# ==================================================
# Spectrogram visualization
# ==================================================

def create_spectrogram_chart(spectrogram):

    n_mels, n_frames = spectrogram.shape

    # Calculate time positions
    frame_times = (
        np.arange(n_frames) * 256 / 22050
    )

    # Convert spectrogram matrix into tabular data
    df = pd.DataFrame({

        "Time": np.tile(
            frame_times,
            n_mels
        ),

        "Mel Band": np.repeat(
            np.arange(n_mels),
            n_frames
        ),

        "Power (dB)": spectrogram.flatten()

    })

    # Calculate rectangle boundaries
    df["Time End"] = np.minimum(
        df["Time"] + 256 / 22050,
        1.0
    )

    df["Mel End"] = (
        df["Mel Band"] + 1
    )

    # Create heatmap
    chart = alt.Chart(

        alt.Data(
            values=df.to_dict("records")
        )

    ).mark_rect().encode(

        x=alt.X(
            "Time:Q",
            title="Time (seconds)",
            scale=alt.Scale(
                domain=[0, 1]
            )
        ),

        x2="Time End:Q",

        y=alt.Y(
            "Mel Band:Q",
            title="Mel Frequency Band",
            scale=alt.Scale(
                domain=[0, 128]
            )
        ),

        y2="Mel End:Q",

        color=alt.Color(
            "Power (dB):Q",
            title="Power (dB)",
            scale=alt.Scale(
                scheme="magma",
                domain=[-80, 0]
            )
        ),

        tooltip=[

            alt.Tooltip(
                "Time:Q",
                format=".3f"
            ),

            alt.Tooltip(
                "Mel Band:Q"
            ),

            alt.Tooltip(
                "Power (dB):Q",
                format=".2f"
            )

        ]

    ).properties(

        height=350,

        title="Log-Mel Spectrogram"

    )

    return chart




# ==================================================
# Full recording overview (visualization only)
# ==================================================

@st.cache_data(max_entries=8)
def get_audio_overview(audio_bytes):
    audio, sr = librosa.load(
        io.BytesIO(audio_bytes), sr=22050, mono=True
    )
    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    duration = len(audio) / sr

    # Preserve min/max amplitude per block instead of plotting
    # every sample. Limit the chart to about 2,000 blocks.
    block_size = max(1, int(np.ceil(len(audio) / 2000)))
    starts = np.arange(0, len(audio), block_size)
    waveform_df = pd.DataFrame({
        "Time": starts / sr,
        "Minimum": np.minimum.reduceat(audio, starts),
        "Maximum": np.maximum.reduceat(audio, starts),
    })

    # Analyze the entire recording continuously. These features
    # are for display only, not for inference.
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    )
    full_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

    # Render the full heatmap as a compact PNG for the browser.
    spectrogram_png = render_full_spectrogram(full_db, duration)

    return waveform_df, duration, spectrogram_png, full_db.shape


def create_waveform_chart(waveform_df, duration):
    return alt.Chart(waveform_df).mark_area(
        color="#5486B8", opacity=0.85
    ).encode(
        x=alt.X("Time:Q", title="Time (seconds)",
                scale=alt.Scale(domain=[0, duration], nice=False)),
        y=alt.Y("Maximum:Q", title="Amplitude"),
        y2="Minimum:Q",
        tooltip=[
            alt.Tooltip("Time:Q", format=".3f"),
            alt.Tooltip("Minimum:Q", format=".3f"),
            alt.Tooltip("Maximum:Q", format=".3f"),
        ],
    ).properties(height=220)


def render_full_spectrogram(spectrogram, duration):
    """Render a static full-duration heatmap without Matplotlib."""
    n_mels, n_frames = spectrogram.shape
    plot_width, plot_height = 1000, 320
    left, top = 90, 22
    right = left + plot_width
    bottom = top + plot_height

    # Reduce only the displayed time columns for long recordings.
    # Max pooling retains strong transient energy in each column.
    if n_frames > plot_width:
        starts = np.arange(plot_width) * n_frames // plot_width
        display_db = np.maximum.reduceat(spectrogram, starts, axis=1)
    else:
        display_db = spectrogram

    # A dark-to-bright, magma-like color palette.
    anchors = np.array([
        [0, 0, 4], [28, 16, 68], [79, 18, 123],
        [129, 37, 129], [181, 54, 122], [229, 80, 100],
        [251, 135, 97], [254, 194, 135], [252, 253, 191],
    ], dtype=float)
    positions = np.linspace(0, 1, len(anchors))
    lut = np.stack([
        np.interp(np.linspace(0, 1, 256), positions, anchors[:, c])
        for c in range(3)
    ], axis=1).astype(np.uint8)

    levels = np.rint(np.clip((display_db + 80) / 80, 0, 1) * 255)
    # Flip the frequency axis: low bands at the bottom.
    rgb = lut[levels.astype(np.uint8)[::-1]]
    heatmap = Image.fromarray(rgb).resize(
        (plot_width, plot_height), Image.Resampling.NEAREST
    )

    canvas = Image.new("RGB", (1260, 420), "white")
    canvas.paste(heatmap, (left, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    small_font = ImageFont.load_default(size=15)
    ink = "#333333"
    draw.rectangle((left, top, right, bottom), outline=ink)

    # Time ticks use the full recording duration (including tail).
    for i in range(7):
        fraction = i / 6
        x = round(left + fraction * plot_width)
        draw.line((x, bottom, x, bottom + 6), fill=ink, width=1)
        draw.text((x, bottom + 10), f"{duration * fraction:.2f}",
                  fill=ink, font=small_font, anchor="mt")
    draw.text(((left + right) / 2, bottom + 42), "Time (seconds)",
              fill=ink, font=font, anchor="mt")

    # The vertical axis is a Mel-band index, not Hz.
    for band in sorted(set([0, 32, 64, 96, n_mels - 1])):
        if band >= n_mels:
            continue
        y = round(top + (n_mels - 1 - band) / (n_mels - 1) * (plot_height - 1))
        draw.line((left - 6, y, left, y), fill=ink, width=1)
        draw.text((left - 10, y), str(band), fill=ink,
                  font=small_font, anchor="rm")
    axis_label = Image.new("RGB", (250, 28), "white")
    ImageDraw.Draw(axis_label).text((125, 14), "Mel Frequency Band",
                                    fill=ink, font=font, anchor="mm")
    canvas.paste(axis_label.rotate(90, expand=True),
                 (8, top + (plot_height - 250) // 2))

    # Colorbar: relative power in dB, from -80 to 0.
    bar_x = right + 26
    bar_indices = np.rint(np.linspace(255, 0, plot_height)).astype(int)
    bar_rgb = np.repeat(lut[bar_indices][:, None, :], 22, axis=1)
    canvas.paste(Image.fromarray(bar_rgb), (bar_x, top))
    draw.text((bar_x + 30, top), "dB", fill=ink, font=small_font, anchor="lt")
    for db in [0, -20, -40, -60, -80]:
        y = round(top + (-db / 80) * (plot_height - 1))
        draw.text((bar_x + 30, y + (14 if db == 0 else 0)), str(db),
                  fill=ink, font=small_font, anchor="lm")

    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


# ==================================================
# Segment inference and recording-level aggregation
# ==================================================

def predict_segments(model, class_names, spectrograms, batch_size=32):
    """Return all segment probabilities and their recording-level mean."""
    if spectrograms.ndim != 4 or spectrograms.shape[1:] != (1, 128, 87):
        raise ValueError("Expected CNN input of shape (N, 1, 128, 87).")
    if len(spectrograms) == 0:
        raise ValueError("At least one complete 1-second segment is required.")

    device = next(model.parameters()).device
    model.eval()
    inputs = torch.from_numpy(np.asarray(spectrograms, dtype=np.float32))
    batches = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size].to(device)
            logits = model(batch)
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    segment_probabilities = np.concatenate(batches, axis=0)
    if segment_probabilities.shape != (len(inputs), len(class_names)):
        raise ValueError("Model output does not match the segment/class counts.")
    if not np.isfinite(segment_probabilities).all():
        raise ValueError("Model returned non-finite probabilities.")

    # Preserve the existing recording-level rule: average probabilities,
    # not predicted labels or logits.
    probabilities = segment_probabilities.mean(axis=0)
    best = int(np.argmax(probabilities))
    return {
        "class_names": list(class_names),
        "technique": class_names[best],
        "confidence": float(probabilities[best]),
        "probabilities": probabilities,
        "segment_probabilities": segment_probabilities,
    }


def create_segment_results(result):
    segment_probs = result["segment_probabilities"]
    class_names = result["class_names"]
    winners = np.argmax(segment_probs, axis=1)
    n = len(segment_probs)
    return pd.DataFrame({
        "Segment": np.arange(1, n + 1),
        "Start (s)": np.arange(n),
        "End (s)": np.arange(1, n + 1),
        "Technique": [class_names[i].title() for i in winners],
        "Confidence (%)": segment_probs[np.arange(n), winners] * 100,
    })


def create_technique_timeline(segment_df, class_names):
    # Stable colors for each technique, including when a class is absent.
    names = [name.title() for name in class_names]
    return alt.Chart(
        alt.Data(values=segment_df.to_dict("records"))
    ).mark_bar(size=34).encode(
        x=alt.X("Start (s):Q", title="Time (seconds)",
                scale=alt.Scale(domain=[0, len(segment_df)], nice=False)),
        x2=alt.X2("End (s):Q"),
        y=alt.value(25),
        color=alt.Color("Technique:N", title="Predicted technique",
                        scale=alt.Scale(domain=names, scheme="tableau10"),
                        legend=alt.Legend(orient="bottom", columns=3)),
        tooltip=[
            alt.Tooltip("Segment:Q", format="d"),
            alt.Tooltip("Start (s):Q", format=".0f"),
            alt.Tooltip("End (s):Q", format=".0f"),
            alt.Tooltip("Technique:N"),
            alt.Tooltip("Confidence (%):Q", format=".2f"),
        ],
    ).properties(height=55)


# ==================================================
# 4. Page title
# ==================================================

st.title("🎸 Guitar AI")

st.subheader(
    "Electric Guitar Playing Technique Classification"
)

st.write(
    "Upload an electric guitar recording "
    "to identify its playing technique "
    "using a trained convolutional neural network."
)

st.divider()


# ==================================================
# 5. Model information
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Playing Techniques",
        "9"
    )

with col2:

    st.metric(
        "Test Accuracy",
        "84.07%"
    )

with col3:

    st.metric(
        "Macro F1",
        "80.45%"
    )

with col4:

    st.metric(
        "Model",
        "CNN V4"
    )

st.caption(
    "Accuracy and Macro F1 are mean "
    "recording-level results across "
    "the five official test folds."
)

st.divider()


# ==================================================
# 6. Sidebar
# ==================================================

st.sidebar.title("About the Model")

st.sidebar.write(
    "CNN V4 with three convolutional blocks, "
    "Batch Normalization and "
    "Weighted CrossEntropyLoss."
)

st.sidebar.markdown(
    """
    ### Audio Processing

    - Sample rate: 22050 Hz
    - Segment duration: 1 second
    - Mel bands: 128
    - FFT size: 1024
    - Hop length: 256

    ### Dataset

    - 549 original recordings
    - 6044 audio segments
    - 9 playing technique classes

    ### Evaluation

    - Five official folds
    - Recording Accuracy: 84.07%
    - Macro F1: 80.45%
    """
)


# ==================================================
# 7. Upload audio
# ==================================================

st.header("Upload Guitar Recording")

uploaded_file = st.file_uploader(
    "Choose a WAV audio file",
    type=["wav"]
)


# ==================================================
# 8. Analyze uploaded audio
# ==================================================

if uploaded_file is not None:

    audio_bytes = uploaded_file.getvalue()
    audio_id = hashlib.sha256(audio_bytes).hexdigest()

    # A new recording must never display predictions from an older one.
    if st.session_state.get("analysis_audio_id") != audio_id:
        st.session_state["analysis_audio_id"] = audio_id
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("cnn_segment_selector", None)

    spectrograms = None
    selected_segment = 1

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Audio playback
    st.audio(
        audio_bytes,
        format="audio/wav"
    )


    # ======================================
    # Full recording waveform and spectrogram
    # ======================================

    st.subheader("Audio Waveform")

    try:
        with st.spinner("Preparing audio overview..."):
            waveform_df, duration, full_png, full_shape = get_audio_overview(
                audio_bytes
            )

        st.caption(f"Audio Duration: {duration:.2f} seconds")
        st.altair_chart(
            create_waveform_chart(waveform_df, duration),
            use_container_width=True
        )

        st.subheader("Full Audio Spectrogram")
        st.caption(f"Full Log-Mel shape: {full_shape}")
        st.image(full_png, width="stretch")
        st.caption(
            "Continuous full-recording Log-Mel spectrogram. "
            "Power is relative to the recording's peak; "
            "long recordings are compressed in time for display only. "
            "This overview is not the CNN input."
        )

    except Exception as error:
        st.warning(f"Audio overview failed: {error}")

    st.divider()

    # ======================================
    # Display actual CNN input spectrogram
    # ======================================

    st.subheader("Selected CNN Input — Log-Mel Spectrogram")

    try:

        spectrograms = get_spectrograms(
            audio_bytes
        )

        # Number of available segments
        num_segments = len(spectrograms)

        st.caption(
            f"Number of audio segments: "
            f"{num_segments}"
        )

        st.caption(
            f"CNN input shape: "
            f"{spectrograms.shape}"
        )

        # Select an audio segment
        if num_segments > 1:

            selected_segment = st.slider(
                "Choose Audio Segment",
                min_value=1,
                max_value=num_segments,
                value=1,
                step=1,
                key="cnn_segment_selector"
            )

        else:

            selected_segment = 1

        # Convert segment number to array index
        segment_index = selected_segment - 1

        selected_spectrogram = spectrograms[
            segment_index, 0
        ]

        # Display the corresponding time range
        start_time = segment_index
        end_time = start_time + 1

        st.caption(
            f"Displaying Segment "
            f"{selected_segment}: "
            f"{start_time}–{end_time} seconds"
        )

        chart = create_spectrogram_chart(
            selected_spectrogram
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )

    except Exception as error:

        st.warning(
            f"Spectrogram visualization failed: "
            f"{error}"
        )


    analyze_button = st.button(
        "Analyze Playing Technique",
        type="primary",
        use_container_width=True,
        disabled=spectrograms is None or len(spectrograms) == 0,
    )

    if analyze_button:
        st.session_state.pop("analysis_result", None)
        try:
            with st.spinner("Analyzing guitar recording..."):
                model, class_names = get_model()
                st.session_state["analysis_result"] = predict_segments(
                    model, class_names, spectrograms
                )
        except Exception as error:
            st.error(f"Analysis failed: {error}")

    # Render outside the button condition so slider changes retain results.
    result = st.session_state.get("analysis_result")
    if result is not None:
        technique = result["technique"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        class_names = result["class_names"]

        st.divider()
        st.header("Prediction Result")
        result_col, confidence_col = st.columns([2, 1])
        with result_col:
            st.success(technique.title())
        with confidence_col:
            st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(float(np.clip(confidence, 0, 1)))
        st.caption(
            "Recording-level prediction uses the mean of all complete "
            "1-second segment softmax probabilities. Confidence is not "
            "a guaranteed probability of a correct prediction."
        )

        st.header("Class Probabilities")
        results_df = pd.DataFrame({
            "Technique": [name.title() for name in class_names],
            "Probability (%)": probabilities * 100,
        }).sort_values(by="Probability (%)", ascending=False)
        st.bar_chart(results_df.set_index("Technique"))
        st.dataframe(
            results_df, hide_index=True, use_container_width=True,
            column_config={
                "Probability (%)": st.column_config.NumberColumn(format="%.2f%%")
            },
        )

        st.header("Top 3 Predictions")
        for rank, idx in enumerate(np.argsort(probabilities)[::-1][:3], start=1):
            st.write(
                f"**{rank}. {class_names[idx].title()}** — "
                f"{probabilities[idx] * 100:.2f}%"
            )

        st.divider()
        st.header("Segment Predictions")
        segment_df = create_segment_results(result)
        st.caption(
            f"Analyzed {len(segment_df)} complete 1-second segments. "
            "Any remaining audio shorter than one second is excluded from "
            "CNN inference. Each segment is assigned one of the nine "
            "techniques; silence and mixed techniques are not separate classes."
        )
        st.subheader("Technique Timeline")
        st.altair_chart(
            create_technique_timeline(segment_df, class_names),
            use_container_width=True,
        )
        st.caption(
            "Colors indicate predicted labels for fixed windows, not exact "
            "technique onset or offset times."
        )
        st.dataframe(
            segment_df, hide_index=True, use_container_width=True,
            column_config={
                "Confidence (%)": st.column_config.NumberColumn(format="%.2f%%")
            },
        )

        st.subheader("Selected Segment Prediction")
        row = segment_df.iloc[selected_segment - 1]
        st.write(
            f"**Segment {selected_segment} | "
            f"{row['Start (s)']:.0f}–{row['End (s)']:.0f} seconds**"
        )
        seg_col, seg_conf_col = st.columns([2, 1])
        with seg_col:
            st.success(row["Technique"])
        with seg_conf_col:
            st.metric("Segment Confidence", f"{row['Confidence (%)']:.2f}%")
        selected_probs = result["segment_probabilities"][selected_segment - 1]
        selected_df = pd.DataFrame({
            "Technique": [name.title() for name in class_names],
            "Probability (%)": selected_probs * 100,
        }).sort_values(by="Probability (%)", ascending=False)
        st.bar_chart(selected_df.set_index("Technique"))

else:
    st.session_state.pop("analysis_audio_id", None)
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("cnn_segment_selector", None)
    st.info("Upload a guitar recording to begin classification.")


# ==================================================
# 12. Footer
# ==================================================

st.divider()

st.caption(
    "AIST2010 Course Project | "
    "Electric Guitar Playing Technique Classification"
)
