import torch.nn as nn


class PhotonArivalTimePars(nn.Module):
    """
    MLP for predicting the parameters of the photon arrival time distribution.
    """

    def __init__(self, n_per_layer, input_size, output_size, dropout=0.5):

        super().__init__()

        layers = [
            nn.Linear(input_size, n_per_layer[0]),
            nn.BatchNorm1d(n_per_layer[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        for i in range(1, len(n_per_layer)):
            layers += [
                nn.Linear(n_per_layer[i - 1], n_per_layer[i]),
                nn.BatchNorm1d(n_per_layer[i]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(n_per_layer[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
