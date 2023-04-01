from typing import Tuple

import cv2
import numpy as np


def draw_bb(
    image: np.array, bb: np.array, color: Tuple[int, int, int] = (0, 255, 0)
) -> np.array:
    """
    Draw a bounding box on an image
    Args:
        image: image to draw on
        bb: bounding box (cx, cy, w, h)
        color: color of the bounding box
    """
    return cv2.rectangle(
        # from (C, H, W) to (H, W, C)
        image,
        (round(bb[0] - bb[2] / 2), round(bb[1] - bb[3] / 2)),
        (round(bb[0] + bb[2] / 2), round(bb[1] + bb[3] / 2)),
        color,
        1,
    )
