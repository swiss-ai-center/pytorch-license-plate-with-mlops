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

            true_number_idxs, true_canton_idxs = (
                vlabels[:, :-1],
                vlabels[:, -1],
            )
            true_number_idxs = true_number_idxs.cpu().numpy().astype(int)
            pred_number, pred_canton = model(vinputs)

            # Decode the prediction index
            _, max_number_pred_idx = torch.max(pred_number, dim=2)
            _, max_canton_pred_idx = torch.max(pred_canton, dim=1)
            for i in range(vinputs.shape[0]):
                if i >= max_samples:
                    break
                number = "".join(
                    [
                        str(d)
                        for d in true_number_idxs[i]
                        if d != params.OCRParams.GRU_BLANK_CLASS
                    ]
                )
                number_pred = "".join(
                    [
                        str(d)
                        for d, _ in groupby(
                            max_number_pred_idx[i].cpu().numpy().astype(int)
                        )
                        if d != params.OCRParams.GRU_BLANK_CLASS
                    ]
                )

                canton = list(Canton)[true_canton_idxs[i]].value
                canton_pred = list(Canton)[max_canton_pred_idx[i]].value
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
