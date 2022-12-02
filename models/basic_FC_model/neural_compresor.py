import torch.nn as nn


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

    def forward(self, x):
        """
        :param x: a tensor with the shape of (b, d)
        :return: a tuple of x_reconstruct with the shape of (b, d) and output of quantizer
        """
        # Encoding
        z = self.encoder(x)     # (b, code_dim)

        # Quantization
        quantizer_out = self.quantizer(z)   # [(b, codebook_dim, code_dim), (b, codebook_dim, code_dim), (b, code_dim)]

        # Decoding
        x_reconstruct = self.decoder(quantizer_out[2])      # (b, d)

        return x_reconstruct, quantizer_out
