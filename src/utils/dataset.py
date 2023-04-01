import os
from typing import Tuple, Union

import numpy as np
import torch
from dvclive import Live
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from src.utils.seed import seed_worker


def split_dataset(
    dataset: Dataset, train_split: float
) -> Tuple[DataLoader, DataLoader]:
    train_size = round(train_split * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def get_data_loader(
    dataset: Dataset, multiprocessing: bool, **kwargs
) -> DataLoader:
    assert (
        "shuffle" not in kwargs
    ), "shuffle is not supported for reproducibility"

    multiprocessing_args = {}
    if multiprocessing:
        multiprocessing_args = {
            "num_workers": 2,
            "pin_memory": True,
        }

    # g = torch.Generator()
    # g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        **multiprocessing_args,
        **kwargs,
        worker_init_fn=seed_worker,
        # generator=g,
    )

    loader

    return loader


def save_dataloader(data_loader: DataLoader, path: str, filename: str) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(data_loader, os.path.join(path, filename))


def print_batch_features(
    minibatch: Union[Tensor, list], labels: Tuple[str]
) -> None:
    for batch, label in zip(minibatch, labels):
        print(f"{label} batch shape: {batch.size()}")
        print(f"{label} batch sample:")
        print(f"  shape: {batch[0].shape}")
        print(f"  dtype: {batch[0].dtype}")
        print(f"  min: {np.min(batch.numpy())}")
        print(f"  max: {np.max(batch.numpy())}")
        print(f"  mean: {np.mean(batch.numpy())}")


def save_samples(
    minibatch: Union[Tensor, list],
    path: str,
    n_samples: int = 5,
) -> None:
    imgs, _ = minibatch
    with Live(path) as live:
        for i, img in enumerate(imgs[:n_samples]):
            img = img.numpy().transpose(1, 2, 0) * 255
            img = img.astype(np.uint8)
            if img.shape[2] == 1:
                img = np.squeeze(img, axis=2)
                img = Image.fromarray(img, mode="L")
            else:
                img = Image.fromarray(img)
            live.log_image(f"sample_{i}.png", img)
