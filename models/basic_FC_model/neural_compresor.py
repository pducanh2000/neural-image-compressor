import torch
import torch.nn as nn

# from .encoder import Encoder
# from .decoder import Decoder
# from .quantizer import Quantizer


class NeuralCompressor(nn.Module):
    def __init__(self, encoder, decoder, entropy_encoding, quantizer, beta=1., detaching=False):
        super(NeuralCompressor, self).__init__()

        # All the Submodules
        self.encoder = encoder
        self.decoder = decoder
        self.entropy_encoding = entropy_encoding
        self.quantizer = quantizer
        # beta determines how strongly we focus on compression against reconstruction quality
        self.beta = beta
        # We can detach inputs to the rate, then we learn rate and distortion separately
        self.detaching = detaching

    def forward(self, x, reduction="avg"):
        # Encoding
        z = self.encoder(x)

        # Quantization
        quantizer_out = self.quantizer(z)

        # Decoding
        x_reconstruct = self.decoder(quantizer_out[2])

        # Distortion
        distortion = torch.mean(torch.pow(x - x_reconstruct, 2), 1)
        # Rate: entropy encoding
        rate = torch.mean(self.entropy_encoding(quantizer_out[0], quantizer_out[2]), 1)

        # Objective
        objective = distortion + self.beta * rate

        if reduction == 'sum':
            return objective.sum(), distortion.sum(), rate.sum()
        else:
            return objective.mean(), distortion.mean(), rate.mean()
