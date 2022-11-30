import torch
from sklearn.datasets import load_digits

from torch.utils.data import Dataset, DataLoader


class DigitsDataset(Dataset):
    def __init__(self, transform=None):
        super(DigitsDataset, self).__init__()
        self.data = data