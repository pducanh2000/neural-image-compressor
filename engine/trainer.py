import tqdm
from torch.optim import Adam


def train_on_epoch(model, loss_fn, optimizer, train_loader):
    model.train()
    device = next(model.parameters()).device
    for idx, (image, label) in tqdm(enumerate(train_loader)):
        image = image.to(device)
        image_reconstruct, quantize_output = model(image)
        distortion, rate, objective_loss = loss_fn(image, image_reconstruct, quantize_output)
