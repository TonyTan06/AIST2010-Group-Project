# AIST2010 Group Project

## Electric Guitar Playing Technique Classification

A machine learning system for automatically recognizing nine electric guitar playing techniques using Log-Mel Spectrograms and Convolutional Neural Networks (CNNs).

## Project Results

| Model | Mean Test Accuracy | Mean Macro F1 |
|---|---:|---:|
| SVM Baseline | 64.81% | 59.49% |
| CNN V4 | 84.07% | 80.45% |

Results are based on recording-level evaluation across the five official dataset splits.

## Supported Playing Techniques

- Alternate Picking
- Bend
- Hammer-on
- Legato
- Pull-off
- Slide
- Sweep Picking
- Tapping
- Vibrato

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TonyTan06/AIST2010-Group-Project.git
cd AIST2010-Group-Project/guitar_style_dataset
```

### 2. Install Dependencies

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install -r requirements-cnn.txt
```

For Windows, activate the virtual environment using:

```powershell
.\.venv-cnn\Scripts\Activate.ps1
```

Platform-specific dependencies may require adjustments.

### 3. Run Audio Classification

```bash
python cnn_experiments/predict_audio.py "path/to/audio.wav"
```

The program outputs the predicted playing technique and the probability scores for all nine classes.

Pretrained CNN models are included in `results/cnn_v4_5fold/`. Retraining is not required for inference.

### 4. Run the Web Demo

```bash
python -m streamlit run cnn_experiments/app.py
```

Open http://localhost:8501 in your browser.

The Web Demo supports WAV audio upload, playback, technique prediction and class probability visualization. This feature is currently under development.

## Dataset

The project uses the Guitar Style Dataset:

https://zenodo.org/records/10075352

- 549 original audio recordings
- 6,044 one-second audio segments
- Nine playing technique classes

The original audio dataset and generated training dataset are not included in this repository.

## Model Architecture

CNN V4 consists of:

- Log-Mel Spectrogram input: 1 × 128 × 87
- Three convolutional blocks
- Batch Normalization
- Max Pooling and Average Pooling
- Fully connected classification layers
- Weighted CrossEntropyLoss during training

## Reproduce Training

Download and extract the original WAV files into `data/wav/`.

```bash
python cnn_experiments/prepare_dataset.py
python cnn_experiments/train_cnn_5fold.py
```

The training script saves model checkpoints, training histories, fold results and confusion matrices.

## Project Status

- [x] Dataset preparation
- [x] SVM baseline
- [x] CNN model development
- [x] Five-fold evaluation
- [x] Command-line audio prediction
- [ ] Web Demo testing
- [ ] Spectrogram visualization
- [ ] Real-time audio classification
- [ ] External recording evaluation

## Acknowledgements

Original Guitar Style Dataset and baseline implementation:

https://github.com/magcil/guitar_style_dataset

This repository contains additional CNN experiments and application development for the AIST2010 group project. Original source code and dataset materials remain subject to their respective licenses.