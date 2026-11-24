"""Training-only Log-Mel spectrogram augmentation for Experiment 1.

The default ``mode='combined'`` preserves the original augmentation logic.
For an ablation, select one of ``frequency``, ``time`` or ``eq``.

The outer augmentation probability is 0.7. Each enabled transform has
an independent 0.5 probability after that gate, so a single-transform
ablation applies its transform to about 35% of examples on average.
"""

import random

import torch


def augment_logmel(
    inputs,
    probability=0.7,
    max_freq_mask=8,
    max_time_mask=6,
    max_eq_db=2.5,
    mode="combined",
):
    """Augment a (B, 1, n_mels, n_frames) Log-Mel batch.

    Modes:
        combined: frequency masking, time masking and EQ variation
        frequency: frequency masking only
        time: time masking only
        eq: EQ variation only

    Inputs are expected to be approximately in [-80, 0] dB.
    The original input tensor is not modified.
    Apply this function to training batches ONLY.
    """
    supported_modes = ("combined", "frequency", "time", "eq")
    if mode not in supported_modes:
        raise ValueError(
            f"Unknown augmentation mode: {mode!r}. "
            f"Choose from {supported_modes}."
        )

    if inputs.ndim != 4:
        raise ValueError("Expected 4D input: (B, 1, M, T)")
    if inputs.shape[1] != 1:
        raise ValueError("Expected a single spectrogram channel")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if max_freq_mask < 1 or max_time_mask < 1 or max_eq_db < 0:
        raise ValueError("Mask widths must be positive and max_eq_db nonnegative")

    augmented = inputs.clone()
    batch_size, _, n_mels, n_frames = augmented.shape

    for i in range(batch_size):
        if random.random() > probability:
            continue

        # 1. Frequency Masking (original transform probability: 0.5)
        if mode in ("combined", "frequency") and random.random() < 0.5:
            width = random.randint(1, min(max_freq_mask, n_mels))
            start = random.randint(0, n_mels - width)
            augmented[i, 0, start:start + width, :] = -80.0

        # 2. Time Masking (original transform probability: 0.5)
        if mode in ("combined", "time") and random.random() < 0.5:
            width = random.randint(1, min(max_time_mask, n_frames))
            start = random.randint(0, n_frames - width)
            augmented[i, 0, :, start:start + width] = -80.0

        # 3. Mild EQ Variation (original transform probability: 0.5)
        if mode in ("combined", "eq") and random.random() < 0.5:
            slope = random.uniform(-max_eq_db, max_eq_db)
            eq_curve = torch.linspace(
                -1.0,
                1.0,
                n_mels,
                device=augmented.device,
                dtype=augmented.dtype,
            ).view(n_mels, 1)
            augmented[i, 0] += slope * eq_curve
            augmented[i, 0].clamp_(-80.0, 0.0)

    return augmented
