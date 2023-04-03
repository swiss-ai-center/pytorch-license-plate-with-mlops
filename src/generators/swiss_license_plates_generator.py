import random
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import numpy as np

from src.utils import linalg


class PlatePosition(Enum):
    FRONT = "front"
    BACK = "back"


class Canton(Enum):
    # TODO: Add the missing canton
    AI = "ai"
    AG = "ag"
    AR = "ar"
    BE = "be"
    BL = "bl"
    BS = "bs"
    FR = "fr"
    GE = "ge"
    GL = "gl"
    GR = "gr"
    JU = "ju"
    LU = "lu"
    NE = "ne"
    NW = "nw"
    OW = "ow"
    SG = "sg"
    SH = "sh"
    SO = "so"
    SZ = "sz"
    TG = "tg"
    TI = "ti"
    UR = "ur"
    VD = "vd"
    VS = "vs"
    ZG = "zg"
    ZH = "zh"


class SwissLicensePlatesGenerator:
    def __init__(self, template_path: str) -> None:
        self._template_path = template_path
        self._digits_path = f"{self._template_path}/digits"
        self._canton_path = f"{self._template_path}/cantons"

        self._blank_plate = cv2.imread(
            f"{self._template_path}/blank_plate.png"
        )
        self._dot = cv2.imread(f"{self._template_path}/dot.png")
        self._swiss_flag = cv2.imread(f"{self._template_path}/swiss_flag.png")
        self._digits = {}
        for digit in range(10):
            self._digits[digit] = cv2.imread(
                f"{self._digits_path}/{digit}.png"
            )

    def _get_number_img(self, number: str, padding: int) -> np.ndarray:
        """
        Create an image of the number
        Args:
            number: The number of the license plate
            padding: The extra space between each digit in pixels
        Returns:
            image: Image of the number
        """
        length = len(number)
        width = length * 18 + (length - 1) * padding
        img = np.ones((36, width, 3), np.uint8) * 240
        for i, digit in enumerate(number):
            # Every digit is 18x36
            img[
                :, i * 18 + i * padding : i * 18 + i * padding + 18
            ] = self._digits[int(digit)]
        return img

    def _add_dot(self, img: np.ndarray, padding: int) -> np.ndarray:
        """
        Add the dot to the start of the image
        Args:
            img: The image of the number
            padding: The extra space between the dot
        Returns:
            image: Image of the number with dot
        """
        new_img = (
            np.ones(
                (
                    img.shape[0],
                    img.shape[1] + self._dot.shape[1] + padding * 2,
                    3,
                ),
                np.uint8,
            )
            * 240
        )
        new_img[:, padding : self._dot.shape[1] + padding] = self._dot
        new_img[:, self._dot.shape[1] + padding * 2 :] = img
        return new_img

    def _add_canton(
        self,
        img: np.ndarray,
        canton: Canton,
        add_flag: bool,
        flag_padding: int = 8,
    ) -> np.ndarray:
        """
        Add the canton to the image
        Args:
            img: The image of the number
            canton: Canton code
        Returns:
            image: Image of the number with canton
        """
        canton_name = cv2.imread(
            f"{self._canton_path}/names/{canton.value}.png"
        )

        canton_flag_x = 0
        if add_flag:
            canton_flag = cv2.imread(
                f"{self._canton_path}/flags/{canton.value}.png"
            )
            canton_flag_x = flag_padding + canton_flag.shape[1]

        number_start_x = canton_name.shape[1]
        new_img = (
            np.ones(
                (
                    img.shape[0],
                    img.shape[1] + number_start_x + canton_flag_x,
                    3,
                ),
                np.uint8,
            )
            * 240
        )
        # add the number image to the new image
        new_img[:, number_start_x : number_start_x + img.shape[1]] = img
        # add the canton name to the start
        new_img[:, : canton_name.shape[1]] = canton_name
        # add the canton flag to the end
        if add_flag:
            new_img[:, -canton_flag.shape[1] :] = canton_flag

        return new_img

    def _add_swiss_flag(self, img: np.ndarray, padding: int) -> np.ndarray:
        """
        Add the swiss flag to the start of the image
        Args:
            img: The image of the number
            padding: The extra space between the flag
        Returns:
            image: Image of the number with flag
        """
        new_img = (
            np.ones(
                (
                    img.shape[0],
                    img.shape[1] + self._swiss_flag.shape[1] + padding,
                    3,
                ),
                np.uint8,
            )
            * 240
        )
        img_start_x = self._swiss_flag.shape[1] + padding
        new_img[:, img_start_x : img_start_x + img.shape[1]] = img

        new_img[:, : self._swiss_flag.shape[1]] = self._swiss_flag
        return new_img

    def _make_plate(self, img: np.ndarray) -> np.ndarray:
        """
        Add the plate to the image
        Args:
            img: The image of the number
        Returns:
            image: Image of the number with plate
        """
        # blank plate is 235x50
        # img is 36*100
        new_img = self._blank_plate.copy()
        # place the img in the middle of the plate
        x = int((new_img.shape[1] - img.shape[1]) / 2)
        y = int((new_img.shape[0] - img.shape[0]) / 2)
        new_img[y : y + img.shape[0], x : x + img.shape[1]] = img

        return new_img

    def generate_one(
        self,
        canton: Canton,
        number: str,
        position: PlatePosition,
    ) -> np.ndarray:
        """
        Generate a Swiss license plate

        Args:
            canton: Canton code
            number: The number of the license plate
            position: The position of the license plate
        Returns:
            image: Image of the license plate
        """
        assert 0 < len(number) <= 6, "Number must be between 1 and 6 digits"
        nbr_padding = 2
        dot_padding = 8
        flag_padding = 0
        swiss_flag_padding = 0
        if len(number) == 1:
            flag_padding = 26
            swiss_flag_padding = 26
        elif len(number) == 2:
            flag_padding = 22
            swiss_flag_padding = 22
        elif len(number) == 3:
            flag_padding = 16
            swiss_flag_padding = 16
        elif len(number) == 4:
            flag_padding = 12
            swiss_flag_padding = 12
        elif len(number) == 5:
            dot_padding = 6
            flag_padding = 6
            swiss_flag_padding = 8
        elif len(number) == 6:
            nbr_padding = 1
            dot_padding = 3
            flag_padding = 3
            swiss_flag_padding = 5
        if position == PlatePosition.FRONT:
            nbr_padding += 3
            dot_padding += 3
        img = self._get_number_img(number, nbr_padding)
        img = self._add_dot(img, dot_padding)
        img = self._add_canton(
            img,
            canton,
            add_flag=(position == PlatePosition.BACK),
            flag_padding=flag_padding,
        )
        if position == PlatePosition.BACK:
            img = self._add_swiss_flag(img, swiss_flag_padding)

        return self._make_plate(img)

    def generate_one_random(
        self,
        number_distribution: Dict[int, float] = None,
    ) -> Tuple[np.ndarray, str, Canton]:
        """
        Generate a random Swiss license plate
        Args:
            canton: Canton code
            number_distribution: The distribution of the number of digits
        Returns:
            image: Image of the license plate
            number: The number of the license plate
            canton: The canton of the license plate
        """
        if number_distribution is None:
            number_distribution = {
                1: 0.05,
                2: 0.05,
                3: 0.05,
                4: 0.15,
                5: 0.2,
                6: 0.5,
            }
        canton = random.choice(list(Canton))
        length = random.choices(
            list(number_distribution.keys()),
            weights=number_distribution.values(),
            k=1,
        )[0]
        number = str(random.randint(10 ** (length - 1), 10**length - 1))
        position = random.choice(list(PlatePosition))
        return self.generate_one(canton, number, position), number, canton

    def generate_random(
        self,
        count: int,
        number_distribution: Dict[int, float] = None,
    ) -> List[np.ndarray]:
        """
        Generate multiple random Swiss license plates
        Args:
            count: The number of license plates to generate
            number_distribution: The distribution of the number of digits
        Returns:
            data: Generator of images of the license plates
        """
        yield from (
            self.generate_one_random(number_distribution) for _ in range(count)
        )

    @staticmethod
    def random_transform(
        plate: np.ndarray,
        width: int,
        height: int,
        rotation: bool = True,
        scale: bool = True,
        stretch: bool = True,
        shear: bool = True,
        perspective: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((width, height, 3), dtype=np.uint8)
        transform_mat = np.eye(3, dtype=np.float32)
        plate += 1
        plate = np.clip(plate, 0, 255)

        # random rotation
        if rotation:
            angle = np.radians(random.randint(-20, 20))
            rot_mat = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            transform_mat = np.dot(transform_mat, rot_mat)
        # random scale in percent
        if scale:
            scale = random.randint(30, 60)
            scale_mat = np.array(
                [
                    [scale / 100, 0, 0],
                    [0, scale / 100, 0],
                    [0, 0, 1],
                ]
            )
            transform_mat = np.dot(transform_mat, scale_mat)
        # random stretch
        if stretch:
            stretch = random.randint(-40, 40)
            stretch_mat = np.array(
                [
                    [1, 0, 0],
                    [0, 1 + stretch / 100, 0],
                    [0, 0, 1],
                ]
            )
            transform_mat = np.dot(transform_mat, stretch_mat)
        # random shear
        if shear:
            shear = np.radians(random.randint(-30, 30))
            shear_mat = np.array(
                [
                    [1, -np.sin(shear), 0],
                    [0, np.cos(shear), 0],
                    [0, 0, 1],
                ]
            )
            transform_mat = np.dot(transform_mat, shear_mat)
        # random perspective
        if perspective:
            perspective = np.radians(random.randint(-30, 30))
            perspective_mat = np.array(
                [
                    [1, 0, 0],
                    [np.sin(perspective), 1, 0],
                    [0, 0, 1],
                ]
            )
            transform_mat = np.dot(transform_mat, perspective_mat)

        # place the plate in the center
        x = int((width - plate.shape[1]) / 2)
        y = int((height - plate.shape[0]) / 2)
        mask[y : y + plate.shape[0], x : x + plate.shape[1]] = plate
        p_poly = np.array(
            [
                [x, y],
                [x + plate.shape[1], y],
                [x + plate.shape[1], y + plate.shape[0]],
                [x, y + plate.shape[0]],
            ]
        )

        # apply the transform
        mask = linalg.apply_affine_transform_image(
            mask, transform_mat, center=True
        )
        p_poly = linalg.apply_affine_transform_coords(
            p_poly, transform_mat, origin=(width // 2, height // 2)
        )

        # add alpha channel
        mask = np.concatenate(
            [
                mask,
                np.ones((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
                * 255,
            ],
            axis=2,
        )
        # set the black pixel values to transparent [0, 0, 0, 0]
        mask[mask[:, :, 0] == 0] = [0, 0, 0, 0]

        return mask, p_poly
