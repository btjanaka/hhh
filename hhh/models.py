"""Models for bird audio detection."""
import torch
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
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (16,40,431)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (16,20,215)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (32,20,215)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (32,10,107)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (64,10,107)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (64,5,53)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (128,5,53)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (128,2,26)
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # -> (128,1,1)
            nn.Flatten(),
            nn.Linear(in_features=128,
                      out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x):
        return self.model(x)


class BasicSmallerModel(nn.Module):
    """A basic model based on this article:
    https://medium.com/@mikesmales/sound-classification-using-deep-learning-8bc2aa1990b7

    Takes the input MFCC and gradually uses convolution to make it smaller. The
    last part of the architecture is a 128-node fully connected layer connected
    to a single output node. A sigmoid activation converts this output to the
    probability that the audio recording contains a bird sound.
    """

    def __init__(self):
        super(BasicSmallerModel, self).__init__()

        # Input: (1,6,x)
        # (the adaptive layer allows us to deal with any size x -- the annotated
        #  sizes are for when x = 431)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2586,
                      out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x):
        return self.model(x)


class BabyConvNet(nn.Module):

    def __init__(self):
        super(BabyConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # -> (128,1,1)
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (16,40,431)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (16,20,215)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (32,20,215)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (32,10,107)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (64,10,107)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (64,5,53)
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # -> (128,5,53)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> (128,2,26)
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # -> (128,1,1)
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)


class MultifeatureModel(nn.Module):
    """MultifeatureModel"""

    def __init__(self, preprocessor_types: "sequence of classnames of the feature preprocessors"):
        super(MultifeatureModel, self).__init__()

        self.feature_convs = nn.ModuleList([Net() for Net in preprocessor_types])

        self.combiner = nn.Sequential(
            nn.Linear(in_features=len(preprocessor_types) * 128,
                      out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x: "length-n list of features"):
        z_list = [f(xi) for f, xi in zip(self.feature_convs, x)]
        z = torch.cat(z_list, dim=1)  # (BATCH_SIZE, n*128)
        return self.combiner(z)
