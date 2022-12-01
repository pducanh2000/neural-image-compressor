import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from config.config_hyp import params


class DigitsDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        super(DigitsDataset, self).__init__()
        self.images = image_list
        self.labels = label_list
        self.transform = transform

    def __getitem__(self, item):
        if self.transform:
            image = self.transform(self.images[item])
        else:
            image = self.images[item]
        label = self.labels[item]
        return torch.Tensor(image, dtype=torch.float32), torch.Tensor(label, dtype=torch.float64)

    def __len__(self):
        return len(images)


if __name__ == "__main__":
    digit_dataset = load_digits()
    images = digit_dataset["data"]
    labels = digit_dataset["target"]

    train_imgs, test_val_imgs, train_labels, test_val_labels = train_test_split(
        images, labels,
        train_size=0.7,
        shuffle=True,
        random_state=2000
    )
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        test_val_imgs, test_val_labels,
        test_size=0.5,
        shuffle=True,
        random_state=2000
    )

    train_set = DigitsDataset(train_imgs, train_labels)
    val_set = DigitsDataset(val_imgs, val_labels)
    test_set = DigitsDataset(test_imgs, test_labels)

    train_loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=params["batch_size"], shuffle=True)
