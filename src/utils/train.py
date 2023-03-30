import os

import torch
from dvclive import Live
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import src.utils.model as model_utils
from src.models.abstract_model import AbstractModel


def train_one_epoch(
    train_loader: DataLoader,
    device: torch.device,
    model: AbstractModel,
    optimizer: torch.optim.Optimizer,
    epoch_index: int,
    tb_writer: SummaryWriter,
    batch_size: int,
) -> float:
    # Set the model to training mode
    model.train()

    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data[:2]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = model.loss(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # Save metrics after every batch
        last_loss = running_loss / batch_size  # loss per batch

        # Log every 10 steps (10 batches)
        if i % 10:
            print(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def fit(
    model: AbstractModel,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    batch_size: int,
    log_path: str,
    checkpoint_path: str,
    live: Live,
) -> AbstractModel:
    print(model)
    # Print trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        "Model total trainable parameters:",
        f"{trainable_params:,}".replace(",", "'"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    model.to(device)

    writer = SummaryWriter(os.path.join(log_path, "tensorboard"))
    epoch_index = 0
    best_vloss = None
    best_model_state = None

    for _ in range(epochs):
        print(f"EPOCH {epoch_index + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        avg_loss = train_one_epoch(
            train_loader,
            device,
            model,
            optimizer,
            epoch_index,
            writer,
            batch_size,
        )

        # Log the running loss averaged per batch
        # for both training and validation
        avg_vloss = model_utils.eval(model, val_loader)
        live.make_checkpoint
        live.log_metric("epoch", epoch_index + 1)
        live.log_metric("train/loss", avg_loss)
        live.log_metric("validation/loss", float(avg_vloss))

        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_index + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if best_vloss is None or avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_model_state = model.state_dict()
            os.makedirs(checkpoint_path, exist_ok=True)
            model_path = os.path.join(
                checkpoint_path, f"model_best_{epoch_index + 1}"
            )
            torch.save(model.state_dict(), model_path)

        epoch_index += 1
        live.next_step()

    # Set the model to the best_model_state
    model.load_state_dict(best_model_state)
    return model
