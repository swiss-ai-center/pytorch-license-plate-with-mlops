import os
from typing import Union

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


class LicensePlatesDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_folder: str,
        dataset_metadata_path: str,
        img_shape: list[int, int, int],
        img_transform: transforms.Compose,
        max_images: Union[int, None],
        skiprows: int = 0,
    ):
        """
        Args:
            dataset_path: path to the data directory
            dataset_folder: name of the dataset
            dataset_metadata_path: path to the image metadata
            max_images: maximum number of images to load
            img_shape: image shape
            img_transform: transform to be applied on a sample
        """
        self._dataset_path = dataset_path
        self._dataset_folder = os.path.join(self._dataset_path, dataset_folder)
        self._dataset_metadata_path = os.path.join(
            self._dataset_path, dataset_metadata_path
        )
        self._max_images = max_images
        self._img_shape = img_shape
        self._img_transform = img_transform
        # read the txt file each line is a image path, parse the line until ';'
        # and ignore lines that start with 'noplate'
        self._img_metadata = np.loadtxt(
            self._dataset_metadata_path,
            dtype=str,
            delimiter=";",
            usecols=0,
            skiprows=skiprows,
            max_rows=self._max_images,
            comments="noplate",
        )

    def __len__(self):
        return len(self._img_metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self._dataset_folder, self._img_metadata[idx])
        bb_path = img_path.replace(".jpg", ".mat")
        # image = read_image(img_path, mode=ImageReadMode.GRAY)
        image = Image.open(img_path)

        if self._img_shape[0] == 1:
            image = image.convert("L")

        base_offset = 75
        # TODO: Add random zoom
        x_offset = np.random.randint(-base_offset, base_offset)
        y_offset = np.random.randint(-base_offset, base_offset)

        image_cropped = image.crop(
            (
                x_offset,
                y_offset,
                image.width + x_offset,
                image.height + y_offset,
            )
        )
        image_resized = image_cropped.resize(
            (self._img_shape[1], self._img_shape[2])
        )

        poly = loadmat(bb_path)["corners"]
        # bounding box contains the coordinates of the 4 corners 4x2
        # scale the bounding box according to the crop
        x_scale = image_resized.width / image.width
        y_scale = image_resized.height / image.height
        poly[:, 0] = (poly[:, 0] - x_offset) * x_scale
        poly[:, 1] = (poly[:, 1] - y_offset) * y_scale
        poly[:, 0] /= image_resized.width
        poly[:, 1] /= image_resized.height
        # convert polygon to bounding box
        min_x = np.min(poly[:, 0])
        min_y = np.min(poly[:, 1])
        max_x = np.max(poly[:, 0])
        max_y = np.max(poly[:, 1])
        # center x, center y, width, height
        cx = min_x + (max_x - min_x) / 2
        cy = min_y + (max_y - min_y) / 2
        bw = max_x - min_x
        bh = max_y - min_y

        bb = Tensor(np.array([cx, cy, bw, bh]))
        image_tensor = self._img_transform(image_resized)
        return image_tensor, bb
