import torch
import torch.nn as nn


class GuitarCNN(nn.Module):

    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(

            # Convolution Block 1
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Fixed average pooling for Apple MPS
            nn.AvgPool2d(
                kernel_size=(8, 6),
                stride=(8, 5)
            )
        )
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 4 * 4, 64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):

        # Normalize dB values from [-80, 0] to [0, 1]
        x = (x + 80.0) / 80.0

        # Extract features
        x = self.features(x)

        # Classification
        x = self.classifier(x)

        return x

if __name__ == "__main__":

    # Select GPU or CPU
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # Create model
    model = GuitarCNN(num_classes=9).to(device)

    # Generate 4 simulated Mel Spectrograms
    dummy_input = (
        torch.rand(4, 1, 128, 87) * 80 - 80
    ).to(device)

    # Forward pass
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    # Test backward propagation
    labels = torch.tensor(
        [0, 1, 2, 3],
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, labels)
    loss.backward()

    print("Test loss:", loss.item())
    print("Backward pass successful")