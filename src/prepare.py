import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from datasets.licenseplatesdataset import LicensePlatesDataset


def main():
    glob_params = yaml.safe_load(open("params.yaml"))
    params = glob_params["prepare"]
    img_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
        ]
    )

    dataset = LicensePlatesDataset(
        data_dir=params["data_dir"],
        img_folder=params["img_folder"],
        img_metadata_path=params["img_metadata_path"],
        img_shape=glob_params["img_shape"],
        img_transform=img_transform,
        max_images=params["max_images"],
    )
    train_size = int(params["train_split"] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=glob_params["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=glob_params["batch_size"],
        shuffle=True,
    )

    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print("Feature sample:")
    print(f"  shape: {train_features[0].shape}")
    print(f"  dtype: {train_features[0].dtype}")
    print(f"  min: {np.min(train_features[0].numpy())}")
    print(f"  max: {np.max(train_features[0].numpy())}")
    print(f"Labels batch shape: {train_labels.size()}")
    print("Labels sample:")
    print(f"  shape: {train_labels[0].shape}")
    print(f"  dtype: {train_labels[0].dtype}")
    print(f"  min: {np.min(train_labels[0].numpy())}")
    print(f"  max: {np.max(train_labels[0].numpy())}")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)
    torch.save(train_loader, os.path.join("data", "prepared", "train.pt"))
    torch.save(val_loader, os.path.join("data", "prepared", "val.pt"))

    # show the first 32 images with matplotlib
    # TODO: show this only locally
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(16, 8))
    # for i in range(32):
    #     img = (train_features[i].numpy() * 255).astype(np.uint8)
    #     bb = train_labels[i].numpy() * img.shape[1]
    #     img = draw_bb(img, bb)
    #     plt.subplot(4, 8, i + 1)
    #     plt.imshow(img)
    #     plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    main()
