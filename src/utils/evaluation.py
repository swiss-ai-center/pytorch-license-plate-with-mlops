import cv2
import numpy as np
from dvclive import Live
from torch import Tensor


def live_log_img_pred(
    live: Live,
    filename: str,
    img: Tensor,
    number: str,
    canton: str,
    number_pred: str,
    canton_pred: str,
    width: int = 512,
    height: int = 512,
) -> None:
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    # add 100px padding to the bottom of the image
    img = cv2.copyMakeBorder(
        img,
        0,
        100,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    # 3% of the image width
    x_offset = round(img.shape[1] * 0.03)
    y_offset = round(img.shape[0] * 0.03)
    img = cv2.putText(
        img,
        f"True: {canton.upper()} {number}",
        (x_offset, img.shape[0] - y_offset - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    img = cv2.putText(
        img,
        f"Pred: {canton_pred.upper()} {number_pred}",
        (x_offset, img.shape[0] - y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    live.log_image(filename, img)
