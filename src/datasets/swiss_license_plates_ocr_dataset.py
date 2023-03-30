import numpy as np
from PIL import Image
from torch import LongTensor, Tensor
from torchvision import transforms

import params
from src.datasets.swiss_license_plates_dataset import SwissLicensePlatesDataset
from src.generators.swiss_license_plates_generator import Canton
from src.utils import linalg


class SwissLicensePlateOCRDataset(SwissLicensePlatesDataset):
    def __init__(
        self, img_size_before_crop: tuple[int, int], **kwargs
    ) -> None:
        self._crop_shape = kwargs["img_shape"]

        super().__init__(
            **{
                **kwargs,
                "img_shape": (self._crop_shape[0], *img_size_before_crop),
            }
        )
        self._cantons = list(Canton)
        self._plate_transform = transforms.ToTensor()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        plate, bb, number, canton = super().__getitem__(index)
        plate = plate.numpy().transpose(1, 2, 0) * 255
        bb = bb.numpy()

        # Convert the bounding box to pixel coordinates
        bb *= plate.shape[1]

        # Randomly transform the bounding box
        # Scale matrix from 100% to 110%
        scale_mat = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, np.random.uniform(1, 1.1)],
            ]
        )
        # Skew matrix from -2.5° to 2.5°
        angle = np.radians(np.random.uniform(-2.5, 2.5))
        skew_mat = np.array(
            [
                [1, angle, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        transform_mat = np.dot(scale_mat, skew_mat)
        bb = linalg.apply_affine_transform_coords(
            bb.reshape(2, 2),
            transform_mat,
            origin=(plate.shape[0] // 2, plate.shape[1] // 2),
        ).flatten()

        # Crop the plate
        plate = plate[
            round(bb[1] - bb[3] / 2) : round(bb[1] + bb[3] / 2),
            round(bb[0] - bb[2] / 2) : round(bb[0] + bb[2] / 2),
            :,
        ]

        # squeeze the last dim
        plate = plate.squeeze(axis=2)
        plate_resized = (
            Image.fromarray(plate)
            .resize(
                self._crop_shape[1:3],
                resample=Image.NEAREST,
            )
            .convert("L")
        )

        # Get the labels
        number_arr = np.full(
            (params.OCRParams.MAX_LABEL_LENGTH,),
            fill_value=params.OCRParams.GRU_BLANK_CLASS,
            dtype=np.int64,
        )
        for i, digit in enumerate(number):
            number_arr[i] = int(digit)

        canton_arr = np.array(
            [self._cantons.index(Canton(canton))], dtype=np.int64
        )

        plate_tensor = self._plate_transform(plate_resized)
        return plate_tensor, LongTensor(
            np.concatenate((number_arr, canton_arr))
        )
