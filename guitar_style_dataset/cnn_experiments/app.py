
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from predict_audio import (
    load_model,
    predict_audio
)


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

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    # Audio playback
    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    analyze_button = st.button(
        "Analyze Playing Technique",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        try:

            with st.spinner(
                "Analyzing guitar recording..."
            ):

                # Load model only once
                model, class_names = get_model()

                # Save uploaded audio temporarily
                with tempfile.TemporaryDirectory() as tmp_dir:

                    audio_path = (
                        Path(tmp_dir) / "input.wav"
                    )

                    audio_path.write_bytes(
                        audio_bytes
                    )

                    (
                        technique,
                        confidence,
                        probabilities

                    ) = predict_audio(
                        model,
                        class_names,
                        audio_path
                    )

            # ======================================
            # 9. Prediction result
            # ======================================

            st.divider()

            st.header("Prediction Result")

            result_col, confidence_col = st.columns(
                [2, 1]
            )

            with result_col:

                st.success(
                    technique.title()
                )

            with confidence_col:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.progress(
                float(confidence)
            )

            st.caption(
                "Confidence represents the model's "
                "average softmax probability, "
                "not a guaranteed probability "
                "of a correct prediction."
            )

            # ======================================
            # 10. Probability chart
            # ======================================

            st.header("Class Probabilities")

            results_df = pd.DataFrame({

                "Technique": [
                    name.title()
                    for name in class_names
                ],

                "Probability (%)": (
                    probabilities * 100
                )
            })

            results_df = results_df.sort_values(
                by="Probability (%)",
                ascending=False
            )

            st.bar_chart(
                results_df.set_index(
                    "Technique"
                )
            )

            # Probability table
            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True,
                column_config={

                    "Probability (%)":
                        st.column_config.NumberColumn(
                            format="%.2f%%"
                        )
                }
            )

            # ======================================
            # 11. Top 3 predictions
            # ======================================

            st.header("Top 3 Predictions")

            top_indices = np.argsort(
                probabilities
            )[::-1][:3]

            for rank, idx in enumerate(
                top_indices,
                start=1
            ):

                st.write(
                    f"**{rank}. "
                    f"{class_names[idx].title()}** "
                    f"— "
                    f"{probabilities[idx] * 100:.2f}%"
                )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )

else:

    st.info(
        "Upload a guitar recording "
        "to begin classification."
    )


# ==================================================
# 12. Footer
# ==================================================

st.divider()

st.caption(
    "AIST2010 Course Project | "
    "Electric Guitar Playing Technique Classification"
)
