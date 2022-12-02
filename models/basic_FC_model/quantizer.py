import torch
import torch.nn as nn


class Quantizer(nn.Module):
    def __init__(self, input_dim, codebook_dim, temp=10e7):
        super(Quantizer, self).__init__()
        # Temperature for softmax
        self.temp = temp
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim

        # Init codebook uniformly
        # from -1/codebook_dim to 1/codebook_dim
        self.codebook = nn.Parameter(
            torch.FloatTensor(1, self.codebook_dim,).uniform_(-1 / self.codebook_dim, 1 / self.codebook_dim)
        )

    def indices2codebook(self, indices_onehot):
        """
        :param: indices_onehot: a tensor with the shape of (b, code_dim, codebook_dim)
        :return: a vector with the same shape as inputs and each row is an onehot vector
        """
        return torch.matmul(indices_onehot, self.codebook.t()).squeeze()

    def indices_to_onehot(self, inputs_shape, indices):
        indices_hard = torch.zeros(inputs_shape[0], inputs_shape[1], self.codebook_dim)
        return indices_hard.scatter_(2, indices, 1)

    def forward(self, inputs):
        """
        :param: inputs: a tensor with the shape of (b, code_dim)
        :return: a tuple of indices for quantization and quantized tensor
        """
        inputs_shape = inputs.shape  # (b, code_dim)
        # Repeat inputs
        inputs_repeat = inputs.unsqueeze(2).repeat(1, 1, self.codebook_dim)     # (b, m, codebook_dim)

        # Calculate distances between inputs and codebook
        distances = torch.exp(
            -torch.sqrt(torch.pow(inputs_repeat - self.codebook.unsqueeze(1), 2))
        )    # (b, code_dim, codebook_dim)
        # print("Distance shape: ", distances.shape)  # Comment to hide the shape of distance

        # indices hard
        indices = torch.argmax(distances, dim=2).unsqueeze(2)    # (b, code_dim, 1)
        indices_hard: torch.Tensor = self.indices_to_onehot(inputs_shape, indices)   # (b, code_dim, codebook_dim)

        # indices soft
        indices_soft = torch.softmax(self.temp * distances, -1)     # (b, code_dim, codebook_dim)

        # Get the quantized input
        quantized = self.indices2codebook(indices_soft)     # (b, code_dim)
        return indices_soft, indices_hard, quantized
