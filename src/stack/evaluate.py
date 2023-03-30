import os
from itertools import groupby

import torch
from dvclive import Live
from torchvision import transforms

import params
import src.utils.model as model_utils
from src.generators.swiss_license_plates_generator import Canton
from src.models import model_registry
from src.utils import evaluation as evaluation_utils
from src.utils.seed import set_seed


def main() -> None:
    set_seed(params.EvaluateStackParams.SEED)

    # Load models
    localize_model = model_registry[params.LocalizeModelParams.MODEL_NAME]
    model_utils.load_weights(
        localize_model, params.Glob.get_out_save_path("localize"), "model.pt"
    )
    ocr_model = model_registry[params.OCRModelParams.MODEL_NAME]
    model_utils.load_weights(
        ocr_model, params.Glob.get_out_save_path("ocr"), "model.pt"
    )

    val_loader = torch.load(
        os.path.join(params.Glob.get_prepared_data_path("stack"), "val.pt")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    localize_model.to(device)
    ocr_model.to(device)

    # We don't need gradients on to do reporting
    localize_model.eval()
    ocr_model.eval()

    samples = 10
    val_correct = 0
    val_total = 0

    with Live(params.Glob.get_out_evaluation_path("stack")) as live:
        with torch.no_grad():
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels, numbers_str, cantons_str = vdata

                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)

                # Scale the image to the size of the model
                vinputs_scaled_localize = transforms.functional.resize(
                    vinputs,
                    params.LocalizeParams.IMG_SHAPE[1:3],
                    antialias=True,
                )
                bb = localize_model(vinputs_scaled_localize)

                # Convert the bounding box
                bb = bb.cpu().numpy()
                bb *= params.LocalizeParams.IMG_SHAPE[1]
                bb = bb.round().astype(int)

                # Crop the image to the bounding box
                vinputs_scaled_ocr = torch.Tensor(
                    params.BATCH_SIZE, *params.OCRParams.IMG_SHAPE
                )
                for j in range(vinputs.shape[0]):
                    bb[j] = [1 if x == 0 else x for x in bb[j]]
                    # PyTorch transforms expect (cy, cx, h, w)
                    cropped = transforms.functional.crop(
                        vinputs[j], bb[j][1], bb[j][0], bb[j][3], bb[j][2]
                    )
                    scaled = transforms.functional.resize(
                        cropped,
                        params.OCRParams.IMG_SHAPE[1:3],
                        antialias=True,
                    )
                    vinputs_scaled_ocr[j] = scaled

                # Predict the license plate
                pred_number, pred_canton = ocr_model(
                    vinputs_scaled_ocr.to(device)
                )

                # TODO: Compute the loss
                # vloss = ocr_model.loss((pred_number, pred_canton), vlabels)
                # running_vloss += vloss.numpy()

                # Decode the number prediction
                _, max_number_index = torch.max(pred_number, dim=2)
                _, max_canton_index = torch.max(pred_canton, dim=1)

                for j in range(max_number_index.shape[0]):
                    num_raw_prediction = list(
                        max_number_index[j].cpu().numpy()
                    )
                    number_pred = "".join(
                        [
                            str(d)
                            for d, _ in groupby(num_raw_prediction)
                            if d != params.OCRParams.GRU_BLANK_CLASS
                        ]
                    )
                    # Decode the number prediction
                    canton_pred = list(Canton)[max_canton_index[j]].value
                    if i + j <= samples:
                        img_cropped = transforms.functional.resize(
                            vinputs_scaled_ocr[j],
                            (vinputs[j].shape[1], vinputs[j].shape[2]),
                            antialias=True,
                        )
                        img_stack = torch.cat(
                            (vinputs[j].cpu(), img_cropped), dim=2
                        )
                        evaluation_utils.live_log_img_pred(
                            live,
                            f"stack_pred_{i}.png",
                            img_stack,
                            numbers_str[j],
                            cantons_str[j],
                            number_pred,
                            canton_pred,
                            width=512 * 2,
                            height=512,
                        )
                    if (
                        number_pred == numbers_str[j]
                        and canton_pred == cantons_str[j]
                    ):
                        val_correct += 1
                    val_total += 1

        live.summary["val_acc"] = val_correct / val_total


if __name__ == "__main__":
    main()
