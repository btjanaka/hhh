"""Basic model for bird audio detection."""
import torch.nn as nn


class BasicModel(nn.Module):
    """A basic model based on this article:
    https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7

    Takes the input MFCC and gradually uses convolution to make it smaller. The
    last part of the architecture is a 128-node fully connected layer connected
    to a single output node. A sigmoid activation converts this output to the
    probability that the audio recording contains a bird sound.
    """

    def __init__(self):
        super(BasicModel, self).__init__()

        # Input: (1,40,x)
        # (the adaptive layer allows us to deal with any size x -- the annotated
        #  sizes are for when x = 431)
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
            nn.Linear(in_features=128, out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x):
        # pylint: disable=arguments-differ
        return self.model(x)
