import torch
import torch.nn as nn


class ObjectiveLoss(nn.Module):
    def __init__(self, entropy_encoding, beta, reduction="avg"):
        super(ObjectiveLoss, self).__init__()
        self.entropy_encoding = entropy_encoding
        self.beta = beta
        self.reduction = reduction

    def forward(self, x, x_reconstruct, quantizer_output):
        """
        :param x: image input
        :param x_reconstruct: image after encode-decode process
        :param quantizer_output: output of quantizer (indices soft, indices_hard, quantized)
        :return: distortion, rate, objective loss
        """
        # Distortion
        distortion = torch.mean(torch.pow((x - x_reconstruct), 2), 1)

        # Rate
        rate = torch.mean(self.entropy_encoding(quantizer_output[0], quantizer_output[2]), 1)

        # Objective
        objective = distortion + self.beta * rate

        if self.reduction == "sum":
            return distortion.sum(), rate.sum(), objective.sum()
        else:
            return distortion.mean(), rate.mean(), objective.mean()