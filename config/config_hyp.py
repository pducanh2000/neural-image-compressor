import torch

params = {
    # Paths
    "checkpoint_folder": "./data/checkpoint/",
    "result_folder": ".data/result/",
    "data_folder": ".data/dataset/",

    # Training params
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "entropy_coding_type": "arm",   # ["arm", "indp", "uniform"]
    "batch_size": 32,
    "beta": 1,                      # If entropy_coding_type is uniform, beta should be 0
    "lr": 1e-3,                     # Learning rate
    "num_epochs": 1000,             # max number of epochs
    "max_patience": 50,              # patience for early stopping

    # Size params
    "D": 64,                        # input dimension
    "C": 16,                        # code length
    "E": 8,                         # codebook size - the number of quantized values
    "M": 256,                       # Number of hidden neurons
    "M_kernels": 32,                # the number of kernel in causal conv 1d layers
    "EPS": 1e-7,                    # param in torch clamp to avoid zero
}
