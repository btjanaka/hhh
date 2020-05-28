"""Convolutional nets for preprocessing audio features."""
import torch.nn as nn


class BabyConvNet(nn.Module):
    """A smaller network for inputs that are not large enough for ConvNet.

    (Since ConvNet pools 4 times and thus the image needs to be at least 16 in
    each dimension).
    """

    def __init__(self):
        super(BabyConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # -> (128,1,1)
            nn.Flatten(),
        )

    def forward(self, x):
        # pylint: disable=arguments-differ
        return self.model(x)


class ConvNet(nn.Module):
    """A convolutional network for preprocessing features.

    The output is always a 128-long vector, regardless of input size.
    """

    def __init__(self):
        super(ConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # -> (16,40,431)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (16,20,215)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # -> (32,20,215)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (32,10,107)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # -> (64,10,107)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (64,5,53)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # -> (128,5,53)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (128,2,26)
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # -> (128,1,1)
            nn.Flatten(),
        )

    def forward(self, x):
        # pylint: disable=arguments-differ
        return self.model(x)
