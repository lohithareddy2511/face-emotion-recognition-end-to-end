"""AlexNet-based CNN model for Face Emotion Recognition.

Modified AlexNet architecture adapted for grayscale facial emotion classification.
Key modifications:
- Input channel reduced to 1 (grayscale)
- Output classes set to 7 (emotion categories)
- Batch normalization added for training stability
- Adaptive pooling for flexible input sizes
"""

import torch
import torch.nn as nn


class AlexNetEmotion(nn.Module):
    """AlexNet architecture adapted for facial emotion recognition.

    Architecture:
        - 5 convolutional layers with ReLU activation and batch normalization
        - 3 max pooling layers
        - 3 fully connected layers with dropout
        - Output: 7-class emotion prediction
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1: 1x227x227 -> 96x55x55
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 96x27x27

            # Conv2: 96x27x27 -> 256x27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256x13x13

            # Conv3: 256x13x13 -> 384x13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Conv4: 384x13x13 -> 384x13x13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Conv5: 384x13x13 -> 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256x6x6
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before the final classification layer."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Pass through all but last layer of classifier
        for layer in list(self.classifier.children())[:-1]:
            x = layer(x)
        return x


def get_model(num_classes: int = 7, dropout: float = 0.5, pretrained_path: str = None) -> AlexNetEmotion:
    """Factory function to create and optionally load a pretrained model."""
    model = AlexNetEmotion(num_classes=num_classes, dropout=dropout)
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    return model
