import random
import torch
import argparse
# import cv2
import matplotlib.pyplot as plt
from dataload.dataset import load_digits, DigitsDataset

from config.config_hyp import params as config_params
from models import NeuralCompressor, Encoder, Decoder, Quantizer, ARMNet, ARMEntropyCoding

# Argparse
parser = argparse.ArgumentParser(description='Reconstruct some images.')
# parser.add_argument('--path', type=str, help="Path to image to eval reconstruct")
parser.add_argument('--checkpoint_path', type=str, default="", help="Path to checkpoint")

args = parser.parse_args()
args = vars(args)

# Model
# Layer hyper parameters
d = config_params["D"]      # input dimension
m = config_params["M"]      # number of hidden neurons
c = config_params["C"]      # code length aka code dim
e = config_params["E"]      # codebook size
beta = config_params["beta"]

# Init modules and full model
encoder = Encoder(d, m, c)
decoder = Decoder(d, m, c)
quantizer = Quantizer(input_dim=d, codebook_dim=e)
arm_net = ARMNet(num_kernels=config_params["M_kernels"], kernel_size=4)
entropy_encoding = ARMEntropyCoding(code_dim=c, codebook_dim=e, arm_net=arm_net)
model = NeuralCompressor(encoder, decoder, entropy_encoding, quantizer, beta, detaching=False)

# Device
device = config_params["device"]
model.to(device)

if __name__ == "__main__":
    # Load data
    digit_data = load_digits()
    digit_dataset = DigitsDataset(digit_data["data"], digit_data["target"])

    # Random image
    random_indices = random.sample(range(0, len(digit_dataset)-1), 3)
    random_images, random_labels = digit_dataset.__getitem__(random_indices)

    # Load model
    model = torch.load(args["checkpoint_path"], map_location="cpu")
    quantizer_output = model.quantizer(model.encoder(random_images))
    reconstruct_images = model.decoder(quantizer_output[2])

    # Plot results
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    fig.tight_layout()
    for i in range(3):
        label = int(random_labels.squeeze()[i].detach().numpy())
        axs[i, 0].set_title("Original Image --- " + str(label))
        axs[i, 0].imshow(random_images[i].squeeze().reshape(8, 8).detach().numpy())

        axs[i, 1].set_title("Reconstructed Image --- " + str(label))
        axs[i, 1].imshow(reconstruct_images[i].squeeze().reshape(8, 8).detach().numpy())
    plt.savefig("./images/sample_reconstruct.jpg", bbox_inches="tight")
    plt.show()
