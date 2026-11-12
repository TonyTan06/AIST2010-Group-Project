
import torch
import torch.nn as nn


class GuitarCNN(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.features = nn.Sequential(

            # ======================================
            # Convolution Block 1
            # ======================================

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # ======================================
            # Convolution Block 2
            # ======================================

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # ======================================
            # Convolution Block 3
            # ======================================

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # ======================================
            # Fixed average pooling for Apple MPS
            # Input: 64 x 16 x 10
            # Output: 64 x 4 x 5
            # ======================================

            nn.AvgPool2d(
                kernel_size=(4, 2),
                stride=(4, 2)
            )
        )


        # ==========================================
        # Classification layers
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 4 * 5,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # Normalize Mel Spectrogram
        # Convert approximately [-80, 0] to [0, 1]

        x = (x + 80.0) / 80.0

        # Extract features
        x = self.features(x)

        # Classification
        x = self.classifier(x)

        return x


# ==================================================
# Model test
# ==================================================

if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    model = GuitarCNN(
        num_classes=9
    ).to(device)

    print("\n===== Model Architecture =====")
    print(model)

    # Count trainable parameters
    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        num_parameters
    )

    # Create dummy input
    dummy_input = (
        torch.rand(4, 1, 128, 87)
        * 80 - 80
    ).to(device)

    print(
        "Input shape:",
        dummy_input.shape
    )

    # Forward pass
    output = model(dummy_input)

    print(
        "Output shape:",
        output.shape
    )

    assert output.shape == (4, 9)

    # Backward pass
    labels = torch.tensor(
        [0, 1, 2, 3],
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        output,
        labels
    )

    loss.backward()

    print("Test loss:", loss.item())

    print("Backward pass successful")
