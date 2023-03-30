import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.abstract_model import AbstractModel


def save(model: torch.nn.Module, path: str, filename: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))


def load_weights(model: torch.nn.Module, path: str, filename: str) -> None:
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(
        torch.load(
            os.path.join(path, filename), map_location=torch.device(device)
        )
    )


def eval(
    model: AbstractModel,
    val_loader: DataLoader,
) -> np.float64:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We don't need gradients on to do reporting
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for vdata in val_loader:
            vinputs, vlabels = vdata[:2]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)

            voutputs = model(vinputs)

            loss = model.loss(voutputs, vlabels)
            running_loss += loss.item()

    return running_loss / len(val_loader)
