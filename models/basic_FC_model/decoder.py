import numpy as np
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, d, m, c):
        super(Decoder, self).__init__()

        self.fc_decoder_net = nn.Sequential(
            nn.Linear(c, m//2),
            nn.BatchNorm1d(m//2),
            nn.ReLU(),
            nn.Linear(m//2, m),
            nn.BatchNorm1d(m),
            nn.ReLU(),
            nn.Linear(m, m*2),
            nn.BatchNorm1d(m*2),
            nn.ReLU(),
            nn.Linear(m*2, d),
        )
        self._init_weights()

    def forward(self, x_embed):
        """
        :param x_embed: a Tensor with the shape of (b, code_dim)
        :return: a reconstructed tensor with the shape of (b, d)
        """
        x_construct = self.fc_decoder_net(x_embed)
        return x_construct

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                y = m.in_features
                m.weight.data.normal_(0.0, 1/np.sqrt(y))
                m.bias.data.fill_(0)
