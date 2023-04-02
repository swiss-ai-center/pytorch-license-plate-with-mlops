import os
from itertools import groupby

import torch
from dvclive import Live

import params
import src.utils.evaluation as evaluation_utils
import src.utils.model as model_utils
from src.generators.swiss_license_plates_generator import Canton
from src.models import model_registry
from src.utils.seed import set_seed


def main() -> None:
    set_seed(params.EvaluateOCRParams.SEED)

    model = model_registry[params.OCRModelParams.MODEL_NAME]

    model_utils.load_weights(
        model, params.Glob.get_out_save_path("ocr"), "model.pt"
    )

    val_loader = torch.load(
        os.path.join(params.Glob.get_prepared_data_path("ocr"), "val.pt")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    max_samples = 10

    # We don't need gradients on to do reporting
    model.eval()

    with Live(params.Glob.get_out_evaluation_path("ocr")) as live:
        with torch.no_grad():
            vdata = next(iter(val_loader))
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)

            true_labels_idxs = vlabels.cpu().numpy().astype(int)
            vlabels_pred = model(vinputs)

            # Decode the prediction index
            _, pred_labels_idxs = torch.max(vlabels_pred, dim=2)
            for i in range(vinputs.shape[0]):
                if i >= max_samples:
                    break
                true = "".join(
                    [
                        str(d - len(Canton))
                        if d >= len(Canton)
                        else list(Canton)[d].value
                        for d in true_labels_idxs[i]
                        if d != params.OCRParams.GRU_BLANK_CLASS
                    ]
                )
                pred = "".join(
                    [
                        str(d - len(Canton))
                        if d >= len(Canton)
                        else list(Canton)[d].value
                        for d, _ in groupby(
                            pred_labels_idxs[i].cpu().numpy().astype(int)
                        )
                        if d != params.OCRParams.GRU_BLANK_CLASS
                    ]
                )
                canton, number = true[:2], true[2:]
                canton_pred, number_pred = pred[:2], pred[2:]

                evaluation_utils.live_log_img_pred(
                    live,
                    f"pred_{i}.png",
                    vinputs[i],
                    number,
                    canton,
                    number_pred,
                    canton_pred,
                )


if __name__ == "__main__":
    main()
