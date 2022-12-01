from torch.utils.data import DataLoader
from torch.optim import Adam

from config.config_hyp import params
from dataload.dataset import load_digits, train_test_split, DigitsDataset
from engine.trainer import train
from losses.objective_loss import ObjectiveLoss
from models.basic_FC_model import NeuralCompressor, Encoder, Decoder, Quantizer, ARMEntropyCoding, ARMNet


# Load config
config_params = params
for k, v in config_params.items():
    print("{}: {}".format(k, v))

# Data
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

# Loss and Optimizer
loss_fn = ObjectiveLoss(entropy_encoding, beta, reduction="avg")
optimizer = Adam(model.parameters(), lr=config_params["lr"])

# Transfer to GPU
device = config_params["device"]
model.to(device)

if __name__ == "__main__":
    # Start training
    train("FC_model", model, loss_fn, optimizer, train_loader, val_loader, config_params)
