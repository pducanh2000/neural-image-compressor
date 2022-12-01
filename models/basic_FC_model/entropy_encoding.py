import torch
import torch.nn as nn

from config.config_hyp import params


class UniformEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(UniformEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim

        self.probs = torch.softmax(torch.ones(1, self.code_dim, self.codebook_dim), -1)

    def sample(self, quantizer=None, b=10):
        code = torch.zeros(b, self.code_dim, self.codebook_dim)
        for b_idx in range(b):
            index = torch.multinomial(torch.softmax(self.probs, -1).squeeze(0), 1).squeeze()
            for i in range(self.code_dim):
                code[b_idx, i, index[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(self.probs, params["EPS"], 1. - params["EPS"])
        return -torch.sum(z * torch.log(p), 2)


class IndependentEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim):
        super(IndependentEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim

        self.probs = nn.Parameter(torch.ones(1, self.code_dim, self.codebook_dim))

    def sample(self, quantizer=None, b=10):
        code = torch.zeros(b, self.code_dim, self.codebook_dim)
        for b_idx in range(b):
            index = torch.multinomial(torch.softmax(self.probs, -1).squeeze(0), 1).squeeze()
            for i in range(self.code_dim):
                code[b_idx, i, index[i]] = 1

        code = quantizer.indices2codebook(code)
        return code

    def forward(self, z, x=None):
        p = torch.clamp(torch.softmax(self.probs, -1), params["EPS"], 1. - params["EPS"])
        return -torch.sum(z * torch.log(p), 2)


# Arithmetic Entropy encoding
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, a=False, **kwargs):
        super(CausalConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.a = a
        self.padding = (self.kernel_size - 1) * self.dilation + self.a * 1

        self.conv1D = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
            dilation=self.dilation,
            **kwargs
        )

    def forward(self, x):
        out = torch.nn.functional.pad(x, (self.padding, 0))
        out = self.conv1D(out)
        if self.a:
            return out[:, :, :-1]
        else:
            return out


class ARMNet(nn.Module):
    def __init__(self, num_kernels=params["M_kernels"], kernel_size=4):
        super(ARMNet, self).__init__()
        # it takes b x 1 x code_dim and outputs b x codebook_dim x code_dim
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.arm_net = nn.Sequential(
            CausalConv1D(1, num_kernels, kernel_size=self.kernel_size, a=True, bias=True),
            nn.LeakyReLU(),
            CausalConv1D(self.num_kernels, self.num_kernels, kernel_size=self.kernel_size, a=False, bias=True),
            nn.LeakyReLU(),
            CausalConv1D(self.num_kernels, out_channels=params["E"], kernel_size=kernel_size, a=False, bias=True))

    def forward(self, x):
        h = self.arm_net(x)
        h = h.permute(2, 0, 1)
        p = torch.softmax(h, 2)
        return p


class ARMEntropyCoding(nn.Module):
    def __init__(self, code_dim, codebook_dim, arm_net):
        super(ARMEntropyCoding, self).__init__()
        self.code_dim = code_dim
        self.codebook_dim = codebook_dim
        self.arm_net = arm_net  # it takes b x 1 x code_dim and outputs b x codebook_dim x code_dim

    def f(self, x):
        h = self.arm_net(x.unsqueeze(1))
        h = h.permute(0, 2, 1)
        p = torch.softmax(h, 2)

        return p

    def sample(self, quantizer=None, b=10):
        x_new = torch.zeros((b, self.code_dim))

        for d in range(self.code_dim):
            p = self.f(x_new)
            index_d = torch.multinomial(p[:, d, :], num_samples=1)
            codebook_value = quantizer.codebook[0, index_d].squeeze()
            x_new[:, d] = codebook_value

        return x_new

    def forward(self, z, x):
        p = self.f(x)
        return -torch.sum(z * torch.log(p), 2)
