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
        self.best_val_loss = 9999


def train_on_epoch(model, loss_fn, optimizer, scheduler, train_loader, state):
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
            Obj_Loss=epoch_objective_loss,
            LR=scheduler.get_last_lr()[0]
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


def train(model, loss_fn, optimizer, scheduler, train_loader, val_loader, config_params):
    # Create checkpoint folder
    checkpoint_model_folder_path = os.path.join(config_params["checkpoint_folder"], config_params["model_name"])
    if not os.path.isdir(checkpoint_model_folder_path):
        os.mkdir(checkpoint_model_folder_path)
    else:
        if config_params["resume"]:
            last_save_path = os.path.join(checkpoint_model_folder_path, "last.pth")
            device = next(model.parameters()).device
            model.load_state_dict(torch.load(last_save_path, map_location=device))

    state = State()
    for epoch in range(config_params["num_epochs"]):
        state.epoch = epoch
        print("-"*50 + "Epoch: {}".format(epoch))
        # Train phase
        state = train_on_epoch(
            model,
            loss_fn,
            optimizer,
            scheduler,
            train_loader,
            state
        )

        # Eval phase
        state = eval_on_epoch(model, loss_fn, val_loader, state)

        if state.val_objective_losses[-1] < state.best_val_loss:
            save_path = os.path.join(checkpoint_model_folder_path, "best.pth")
            torch.save(model, save_path)
            state.patience = 0
            state.best_val_loss = state.val_objective_losses[-1]
            print("Saved best model!!!")
        else:
            save_path = os.path.join(checkpoint_model_folder_path, "last.pth")
            torch.save(model, save_path)
            state.patience += 1
            scheduler.step()
            print("Saved last model!!!")
        print("Best Val Loss: ", state.best_val_loss)
        if state.patience >= config_params["max_patience"]:
            break
    return state
