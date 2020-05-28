"""Bird audio detection model which uses multiple features."""
import torch
import torch.nn as nn


class MultifeatureModel(nn.Module):
    """A model which combines multiple features.

    Each feature is preprocessed, and the outputs are combined into a final fully
    connected layer with one output.

    In many ways, this is a generalization of BasicModel.
    """

    def __init__(self, preprocessor_types: "sequence of classnames of the feature preprocessors"):
        super(MultifeatureModel, self).__init__()

        self.feature_preprocessors = nn.ModuleList([Net() for Net in preprocessor_types])

        self.combiner = nn.Sequential(
            nn.Linear(in_features=len(preprocessor_types) * 128,
                      out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
            nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
        )

    def forward(self, x: "length-n list of features"):
        # pylint: disable=arguments-differ
        processed_features = [f(xi) for f, xi in zip(self.feature_preprocessors, x)]
        combined_inputs = torch.cat(processed_features, dim=1)  # (BATCH_SIZE, n*128)
        return self.combiner(combined_inputs)
