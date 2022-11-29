entropy_coding_type = 'arm' # arm or indp or uniform
D = 64   # input dimension
C = 16  # code length
E = 8 # codebook size (i.e., the number of quantized values)
M = 256  # the number of neurons
M_kernels = 32 # the number of kernels in causal conv1d layers

# beta: how much we weight rate
if entropy_coding_type == 'uniform':
    beta = 0.
else:
    beta = 1.

lr = 1e-3 # learning rate
num_epochs = 1000 # max. number of epochs
max_patience = 50 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped
