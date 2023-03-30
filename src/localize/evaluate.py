import os

import cv2
import numpy as np
import torch
from dvclive import Live

import params
import src.utils.model as model_utils
from src.models import model_registry
from src.utils.preview import draw_bb
from src.utils.seed import set_seed


def main() -> None:
    set_seed(params.EvaluateLocalizeParams.SEED)

    model = model_registry[params.LocalizeModelParams.MODEL_NAME]

    model_utils.load_weights(
        model, params.Glob.get_out_save_path("localize"), "model.pt"
    )

    val_loader = torch.load(
        os.path.join(params.Glob.get_prepared_data_path("localize"), "val.pt")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = 10

    # We don't need gradients on to do reporting
    model.eval()

    with Live(params.Glob.get_out_evaluation_path("localize")) as live:
        with torch.no_grad():
            vdata = next(iter(val_loader))
            vinputs, vlabels = vdata[:2]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)

            voutputs = model(vinputs)

            for i in range(vinputs.shape[0]):
                if i >= max_samples:
                    break
                img = vinputs[i].detach().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8).squeeze()
                bb = vlabels[i].detach().numpy() * img.shape[1]
                bb_pred = voutputs[i].detach().numpy() * img.shape[1]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = draw_bb(img, bb, color=(255, 255, 255))
                img = draw_bb(img, bb_pred, color=(0, 255, 0))
                live.log_image(f"pred_{i}.png", img)


if __name__ == "__main__":
    main()
