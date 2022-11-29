import torch
import torch.nn as nn


class UniformEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(UniformEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim

        self.probs = torch.softmax(torch.ones(1, self.code_dim, self.codebook_dim), -1)

    def sample(self, quantizer=None, B=10):
        code = torch.zeros(B, self.code_dim, self.codebook_dim)
        for b in range(B):
            indx = torch.multinomial(torch.softmax(self.probs, -1).squeeze(0), 1).squeeze()
            for i in range(self.code_dim):
                code[b, i, indx[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(self.probs, EPS, 1. - EPS)
        return -torch.sum(z * torch.log(p), 2)


class IndependentEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(IndependentEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim

        self.probs = nn.Parameter(torch.ones(1, self.code_dim, self.codebook_dim))

    def sample(self, quantizer=None, B=10):
        code = torch.zeros(B, self.code_dim, self.codebook_dim)
        for b in range(B):
            indx = torch.multinomial(torch.softmax(self.probs, -1).squeeze(0), 1).squeeze()
            for i in range(self.code_dim):
                code[b, i, indx[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(torch.softmax(self.probs, -1), EPS, 1. - EPS)
        return -torch.sum(z * torch.log(p), 2)


class ARMEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim, arm_net):
        super(ARMEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim
        self.arm_net = arm_net  # it takes B x 1 x code_dim and outputs B x codebook_dim x code_dim

    def f(self, x):
        h = self.arm_net(x.unsqueeze(1))
        h = h.permute(0, 2, 1)
        p = torch.softmax(h, 2)

        return p

    def sample(self, quantizer=None, B=10):
        x_new = torch.zeros((B, self.code_dim))

        for d in range(self.code_dim):
            p = self.f(x_new)
            indx_d = torch.multinomial(p[:, d, :], num_samples=1)
            codebook_value = quantizer.codebook[0, indx_d].squeeze()
            x_new[:, d] = codebook_value

        return x_new

    def forward(self, z, x):
        p = self.f(x)
        return -torch.sum(z * torch.log(p), 2)
