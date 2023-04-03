from itertools import groupby

import numpy as np
import torch
from mlem.api import save
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.io import read_image

import params
import src.utils.model as model_utils
from src.generators.swiss_license_plates_generator import Canton
from src.models import model_registry


class SwissLicensePlateModelStack:
    def __init__(self, locize_model: nn.Module, ocr_model: nn.Module) -> None:
        self.locize_model = locize_model
        self.locize_model.eval()
        self.ocr_model = ocr_model
        self.ocr_model.eval()

        self._transforms_localize = transforms.Compose(
            [
                transforms.Grayscale(params.LocalizeParams.IMG_SHAPE[0]),
                transforms.Resize(
                    params.LocalizeParams.IMG_SHAPE[1:3],
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )
        self._transforms_ocr = transforms.Compose(
            [
                transforms.Grayscale(params.OCRParams.IMG_SHAPE[0]),
                transforms.ToTensor(),
            ]
        )
        self.__name__ = "SwissLicensePlateModelStack"

    def predict(self, img: Tensor, *args, **kwargs) -> dict:
        return img

    def __call__(self, img: Tensor, *args, **kwargs) -> dict:
        img = Image.fromarray(img.numpy().transpose(1, 2, 0).squeeze())
        # convert to rgb if needed
        if img.mode == "RGBA":
            img = img.convert("RGB")

        img_localize = self._transforms_localize(img).unsqueeze(0)
        bb = self.locize_model(img_localize)
        bb = (
            (
                bb.squeeze(0).detach().cpu().numpy()
                * params.PrepareOCRParams.IMG_SIZE[0]
            )
            .round()
            .astype(np.int32)
        )

        img_ocr = self._transforms_ocr(img)
        cx = bb[0]
        cy = bb[1]
        w = bb[2]
        h = bb[3]
        x = round(cx - w / 2)
        y = round(cy - h / 2)
        img_ocr = transforms.functional.crop(img_ocr, y, x, h, w)
        img_ocr = transforms.functional.resize(
            img_ocr,
            (
                params.OCRParams.IMG_SHAPE[1],
                params.OCRParams.IMG_SHAPE[2],
            ),
            antialias=True,
        )
        pred = self.ocr_model(img_ocr.unsqueeze(0))
        _, pred_label_idxs = torch.max(pred.squeeze(0), dim=1)

        pred = "".join(
            [
                str(d - len(Canton))
                if d >= len(Canton)
                else list(Canton)[d].value
                for d, _ in groupby(pred_label_idxs.cpu().numpy())
                if d != params.OCRParams.GRU_BLANK_CLASS
            ]
        )
        canton_pred, number_pred = pred[:2], pred[2:]

        return {
            "plate": pred.upper(),
            "canton": canton_pred,
            "number": number_pred,
        }


localize_model = model_registry[params.LocalizeModelParams.MODEL_NAME]
model_utils.load_weights(
    localize_model, params.Glob.get_out_save_path("localize"), "model.pt"
)
ocr_model = model_registry[params.OCRModelParams.MODEL_NAME]
model_utils.load_weights(
    ocr_model, params.Glob.get_out_save_path("ocr"), "model.pt"
)
model = SwissLicensePlateModelStack(
    locize_model=localize_model, ocr_model=ocr_model
)

img = read_image("sample.png")
save(
    obj=model,
    path="models/model_stack",
    sample_data=img,
)
