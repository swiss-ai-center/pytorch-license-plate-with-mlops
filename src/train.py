import os
from datetime import datetime

import torch
import yaml
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter

from models.vgg16_model import VGG16Model
from utils.seed import set_seed

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def train_one_epoch(
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    epoch_index: int,
    tb_writer: SummaryWriter,
    bactch_size: int,
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # accelerator.backward(loss)

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        # Save metrics after every batch
        if i % bactch_size == bactch_size - 1:
            last_loss = running_loss / bactch_size  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def main():
    glob_params = yaml.safe_load(open("params.yaml"))
    params = glob_params["train"]
    set_seed(params["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # TODO: Add support for multi-GPU training with accelerate
    accelerator = Accelerator()

    model = VGG16Model(
        glob_params["img_shape"],
        glob_params["hidden_layers"],
        glob_params["fc_features_in"],
        glob_params["num_classes"],
        glob_params["dropout"],
    ).to(device)

    print(model)

    # Print trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model total trainable parameters: {trainable_params}")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
    )

    writer = SummaryWriter("runs/trainer_{}".format(timestamp))
    epoch_index = 0

    train_loader = torch.load(os.path.join("data", "prepared", "train.pt"))
    val_loader = torch.load(os.path.join("data", "prepared", "val.pt"))

    # model, optimizer, train_loader, val_loader = accelerator.prepare(
    #     model, optimizer, train_loader, val_loader
    # )

    best_vloss = None

    for _ in range(params["epochs"]):
        print("EPOCH {}:".format(epoch_index + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            train_loader,
            device,
            accelerator,
            model,
            optimizer,
            loss_fn,
            epoch_index,
            writer,
            glob_params["batch_size"],
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_index + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if best_vloss is None or avg_vloss < best_vloss:
            best_vloss = avg_vloss
            os.makedirs(f"models/checkpoints_{timestamp}", exist_ok=True)
            model_path = (
                f"models/checkpoints_{timestamp}/model_{epoch_index + 1}"
            )
            torch.save(model.state_dict(), model_path)

        epoch_index += 1


if __name__ == "__main__":
    main()
