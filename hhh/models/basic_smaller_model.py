"""Smaller basic model for bird audio detection."""
import torch.nn as nn


class BasicSmallerModel(nn.Module):
    """A smaller basic model"""

    def __init__(self):
        super(BasicSmallerModel, self).__init__()

        # Input: (1,6,x)
        # (the adaptive layer allows us to deal with any size x -- the annotated
        #  sizes are for when x = 431)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2586, out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x):
        # pylint: disable=arguments-differ
        return self.model(x)
