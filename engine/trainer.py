import os
from tqdm import tqdm
import torch


class State(object):
    def __init__(self):
        self.epoch = 0

        # Train losses log
        self.train_objective_losses = []
        self.train_distortion_losses = []
        self.train_rate_losses = []

        # Val losses log
        self.val_objective_losses = []
        self.val_distortion_losses = []
        self.val_rate_losses = []
        self.patience = 0
        self.best_val_loss = 0


def train_on_epoch(model, loss_fn, optimizer, train_loader, state):
    model.train()
    device = next(model.parameters()).device
    # Init return value
    distortion_values, rate_values, objective_loss_values = [], [], []
    epoch_distortion, epoch_rate, epoch_objective_loss = 0, 0, 0

    # Loop
    tqdm_bar = tqdm(train_loader, total=len(train_loader))
    for idx, (image, label) in enumerate(tqdm_bar):
        if hasattr(model, 'dequantization'):
            if model.dequantization:
                image = image + torch.rand(image.shape)

        # Forward
        image = image.to(device)
        image_reconstruct, quantize_output = model(image)

        # Calculate Loss
        distortion, rate, objective_loss = loss_fn(image, image_reconstruct, quantize_output)
        distortion_values.append(distortion.item())
        rate_values.append(rate.item())
        objective_loss_values.append(objective_loss.item())

        # Optim
        optimizer.zero_grad()
        objective_loss.backward(retain_graph=True)
        optimizer.step()

        # Display the mean loss update on each iter
        epoch_distortion = sum(distortion_values) / len(distortion_values)
        epoch_rate = sum(rate_values) / len(rate_values)
        epoch_objective_loss = sum(objective_loss_values) / len(objective_loss_values)
        tqdm_bar.set_postfix(
            Epoch=state.epoch,
            Steps=idx,
            D=epoch_distortion,
            R=epoch_rate,
            Obj_Loss=epoch_objective_loss
        )
    state.train_distortion_losses.append(epoch_distortion)
    state.train_rate_losses.append(epoch_rate)
    state.train_objective_losses.append(epoch_objective_loss)
    return state


def eval_on_epoch(model, loss_fn, val_loader, state):
    model.eval()
    device = next(model.parameters()).device

    # Init return values
    distortion_values, rate_values, objective_loss_values = [], [], []
    epoch_distortion, epoch_rate, epoch_objective_loss = 0, 0, 0

    # Loop
    tqdm_bar = tqdm(val_loader, total=len(val_loader))
    for idx, (image, label) in enumerate(tqdm_bar):
        # Forward
        image = image.to(device)
        image_reconstruct, quantizer_output = model(image)

        # Calculate loss
        distortion, rate, objective_loss = loss_fn(image, image_reconstruct, quantizer_output)

        distortion_values.append(distortion.item())
        rate_values.append(rate.item())
        objective_loss_values.append(objective_loss.item())

        # Display the mean loss update on each iter
        epoch_distortion = sum(distortion_values) / len(distortion_values)
        epoch_rate = sum(rate_values) / len(rate_values)
        epoch_objective_loss = sum(objective_loss_values) / len(objective_loss_values)
        tqdm_bar.set_postfix(
            Epoch=state.epoch,
            Steps=idx,
            D=epoch_distortion,
            R=epoch_rate,
            Obj_Loss=epoch_objective_loss
        )
    state.val_distortion_losses.append(epoch_distortion)
    state.val_rate_losses.append(epoch_rate)
    state.val_objective_losses.append(epoch_objective_loss)
    return state


def train(model_name, model, loss_fn, optimizer, train_loader, val_loader, config_params):
    state = State()
    for epoch in range(config_params["num_epochs"]):
        state.epoch = epoch
        print("-"*50 + "Epoch: {}".format(epoch))
        # Train phase
        state = train_on_epoch(
            model,
            loss_fn,
            optimizer,
            train_loader,
            state
        )

        # Eval phase
        state = eval_on_epoch(model, loss_fn, val_loader, state)

        if state.objective_losses[-1] < state.best_val_loss:
            print("Saving best model...")
            torch.save(model, os.path.join(config_params["checkpoint_folder"], model_name + "_best.pth"))
            state.patience = 0
        else:
            torch.save(model, os.path.join(config_params["checkpoint_folder"], model_name + "_last.pth"))
            state.patience += 1
        if state.patience >= config_params["max_patience"]:
            break

    return state
