import os

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from src.generators.swiss_license_plates_generator import (
    SwissLicensePlatesGenerator,
)


class SwissLicensePlatesDataset(Dataset):
    def __init__(
        self,
        template_path: str,
        img_shape: tuple[int, int, int],
        img_transform: transforms.Compose,
        max_images: int | None,
        is_eval: bool = False,
    ):
        """
        Args:
            template_path: The path to the template directory
            img_shape: The image shape
            img_transform: The transform to be applied on a sample
            max_images: The maximum number of images to load
            is_eval: Whether the dataset is used for evaluation
        """
        self._generator = SwissLicensePlatesGenerator(template_path)
        self._img_shape = img_shape
        self._img_transform = img_transform
        self._max_images = max_images
        self._is_eval = is_eval
        self._cifar_dataset = datasets.CIFAR10(
            root=os.path.abspath(os.path.join(template_path, "cifar10")),
            train=True,
            download=True,
        )

    def _cifar_random_transform(
        self,
        img: Image.Image,
    ) -> Image.Image:
        out_size = np.random.randint(20, img.size[0])
        return transforms.Compose(
            [
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.125),
                transforms.RandomPerspective(p=0.125),
                transforms.RandomCrop(out_size),
            ]
        )(img)

    def __len__(self):
        return self._max_images

    def __getitem__(self, idx) -> tuple[Tensor, Tensor, str, str]:
        img_plate, number, canton = self._generator.generate_one_random()
        img_plate = np.array(
            transforms.Compose(
                [
                    transforms.ColorJitter(saturation=0.5, hue=0.3),
                    transforms.GaussianBlur(
                        kernel_size=(5, 9), sigma=(0.1, 0.5)
                    ),
                    transforms.RandomAdjustSharpness(sharpness_factor=0.5),
                ]
            )(Image.fromarray(img_plate))
        )
        # infinte loop over the cifar dataset
        random_cifar_img = self._cifar_dataset[idx % len(self._cifar_dataset)][
            0
        ]

        # s = time.time()
        # random crop
        random_cifar_img = self._cifar_random_transform(random_cifar_img)
        random_cifar_img = np.array(
            random_cifar_img.resize((250, 250), Image.BILINEAR)
        )
        # place random circles, squares, polygons on the random cifar image
        if np.random.rand() < 0.5:
            nbr_circles = np.random.randint(0, 15)
            nbr_rect = np.random.randint(0, 5)
            nbr_poly = np.random.randint(0, 1)
            for _ in range(nbr_circles):
                r = np.random.randint(0, 25)
                x = np.random.randint(0, round(random_cifar_img.shape[0]) - r)
                y = np.random.randint(0, round(random_cifar_img.shape[1]) - r)
                color = np.random.randint(0, 255)
                cv2.circle(
                    random_cifar_img,
                    (x, y),
                    r,
                    (int(color), int(color), int(color)),
                    -1,
                )
            for _ in range(nbr_rect):
                w = np.random.randint(0, 50)
                h = np.random.randint(0, 50)
                x = np.random.randint(0, round(random_cifar_img.shape[0]) - w)
                y = np.random.randint(0, round(random_cifar_img.shape[1]) - h)
                color = np.random.randint(0, 255)
                cv2.rectangle(
                    random_cifar_img,
                    (x, y),
                    (x + w, y + h),
                    (int(color), int(color), int(color)),
                    -1,
                )
            for _ in range(nbr_poly):
                edges = np.random.randint(3, 10)
                pts_x = np.random.randint(0, random_cifar_img.shape[0], edges)
                pts_y = np.random.randint(0, random_cifar_img.shape[1], edges)
                pts = np.array([pts_x, pts_y]).T.reshape((-1, 1, 2))
                color = np.random.randint(0, 255)
                cv2.fillPoly(
                    random_cifar_img,
                    [pts],
                    (int(color), int(color), int(color)),
                )
        # print(f"{'Random crop:':<20}", f"{time.time() - s:0.4f}")

        # s = time.time()
        img_plate_mask, p_poly = SwissLicensePlatesGenerator.random_transform(
            img_plate,
            width=random_cifar_img.shape[1],
            height=random_cifar_img.shape[0],
        )
        # print(f"{'Random transform:':<20}", f"{time.time() - s:0.4f}")

        # s = time.time()
        # Place the plate on the noise image both have the same size
        max_x = random_cifar_img.shape[1] - np.max(p_poly[:, 0])
        max_y = random_cifar_img.shape[0] - np.max(p_poly[:, 1])
        x_offset = np.random.randint(-max_x, max_x) // 2
        y_offset = np.random.randint(-max_y, max_y) // 2
        # Crop the mask with the random offsets
        img_plate_mask = np.array(
            Image.fromarray(img_plate_mask).crop(
                (
                    x_offset,
                    y_offset,
                    x_offset + random_cifar_img.shape[1],
                    y_offset + random_cifar_img.shape[0],
                )
            )
        )
        # Apply the offset to the bounding box
        p_poly[:, 0] -= x_offset
        p_poly[:, 1] -= y_offset

        # Extract the alpha channel from the plate image
        alpha = img_plate_mask[:, :, 3] / 255.0

        # Composite them together using np.where conditional
        random_cifar_img[:, :] = np.where(
            np.dstack((alpha, alpha, alpha)),
            img_plate_mask[:, :, :3],
            random_cifar_img[:, :],
        )
        # print(f"{'Compsite the plate:':<20}", f"{time.time() - s:0.4f}")

        # s = time.time()
        # create noise image
        img_noise = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
        noise_scale = np.random.randint(1, 20)
        img_noise = cv2.blur(img_noise, (noise_scale, noise_scale))
        alpha = np.random.uniform(0.5, 1)
        random_cifar_img = cv2.addWeighted(
            random_cifar_img, alpha, img_noise, 1 - alpha, 0
        )
        # print(f"{'Random noise:':<20}", f"{time.time() - s:0.4f}")
        # s = time.time()
        # Random sharpening
        enhancer = ImageEnhance.Sharpness(Image.fromarray(random_cifar_img))
        sharpness = np.random.uniform(0, 5)
        img_noise = np.array(enhancer.enhance(sharpness))
        # print(f"{'Random sharpening:':<20}", f"{time.time() - s:0.4f}")

        # s = time.time()
        # scale the noise image to sef._img_shape
        img_noise = Image.fromarray(img_noise)
        img_noise_resized = img_noise.resize(
            self._img_shape[1:3], Image.BICUBIC
        )
        # update the bounding box with the resize
        scale_x = img_noise_resized.size[0] / img_noise.size[0]
        scale_y = img_noise_resized.size[1] / img_noise.size[1]
        p_poly[:, 0] *= scale_x
        p_poly[:, 1] *= scale_y

        # convert to grayscale if needed
        if self._img_shape[0] == 1:
            img_noise_resized = img_noise_resized.convert("L")

        # scale the bounding box according to the resize
        p_poly[:, 0] /= img_noise_resized.width
        p_poly[:, 1] /= img_noise_resized.height
        # convert polygon to bounding box
        min_x = np.min(p_poly[:, 0])
        min_y = np.min(p_poly[:, 1])
        max_x = np.max(p_poly[:, 0])
        max_y = np.max(p_poly[:, 1])
        # center x, center y, width, height
        cx = min_x + (max_x - min_x) / 2
        cy = min_y + (max_y - min_y) / 2
        bw = max_x - min_x
        bh = max_y - min_y

        bb = Tensor(np.array([cx, cy, bw, bh])).clamp(0, 1)
        image_tensor = self._img_transform(img_noise_resized)
        # print(f"{'Scale the image:':<20}", f"{time.time() - s:0.4f}")

        return image_tensor, bb, number, canton.value
