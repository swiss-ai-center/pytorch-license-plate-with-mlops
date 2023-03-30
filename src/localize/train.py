import os

import torch
from dvclive import Live

import params
import src.utils.model as model_utils
import src.utils.train as train_utils
from src.models import model_registry
from src.utils.seed import set_seed


def main():
    set_seed(params.TrainLocalizeParams.SEED)

    model = model_registry[params.LocalizeModelParams.MODEL_NAME]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.TrainLocalizeParams.LR,
    )

    path = params.Glob.get_prepared_data_path("localize")
    train_loader = torch.load(os.path.join(path, "train.pt"))
    val_loader = torch.load(os.path.join(path, "val.pt"))

    log_path = params.Glob.get_out_log_path("localize")

    with Live(log_path) as live:
        model = train_utils.fit(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=params.TrainLocalizeParams.EPOCHS,
            batch_size=params.BATCH_SIZE,
            log_path=log_path,
            checkpoint_path=params.Glob.get_out_checkpoint_path("localize"),
            live=live,
        )

    model_utils.save(
        model,
        params.Glob.get_out_save_path("localize"),
        "model.pt",
    )


if __name__ == "__main__":
    main()
