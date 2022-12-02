import numpy as np
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d, m, c):
        super(Encoder, self).__init__()
        self.fc_encoder_net = nn.Sequential(
            nn.Linear(d, m * 2),
            nn.BatchNorm1d(m * 2),
            nn.ReLU(),
            nn.Linear(m * 2, m),
            nn.BatchNorm1d(m),
            nn.ReLU(),
            nn.Linear(m, m//2),
            nn.BatchNorm1d(m//2),
            nn.ReLU(),
            nn.Linear(m//2, c)
        )
        self._init_weights()

    def forward(self, x):
        """
        :param x: Input tensor with the shape of (b, d)
        :return: a tensor with the shape of (b, c)
        """

        x_embed = self.fc_encoder_net(x)
        return x_embed

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                m.weight.data.normal_(0.0, 1/np.sqrt(y))
                m.bias.data.fill_(0)
