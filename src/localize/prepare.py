from torchvision import transforms

import params
import src.utils.dataset as dataset_utils
from src.datasets.swiss_license_plates_dataset import SwissLicensePlatesDataset
from src.utils.seed import set_seed


def main():
    set_seed(params.PrepareLocalizeParams.SEED)

    img_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
        ]
    )

    dataset = SwissLicensePlatesDataset(
        template_path=params.glob_params["dataset_path"],
        img_shape=params.LocalizeParams.IMG_SHAPE,
        img_transform=img_transform,
        max_images=params.PrepareLocalizeParams.MAX_IMAGES,
    )

    train_dataset, val_dataset = dataset_utils.split_dataset(
        dataset, params.PrepareLocalizeParams.TRAIN_SPLIT
    )

    loader_params = {
        "multiprocessing": (not params.IS_LOCAL),
        "batch_size": params.BATCH_SIZE,
    }
    train_loader = dataset_utils.get_data_loader(
        train_dataset,
        **loader_params,
    )
    val_loader = dataset_utils.get_data_loader(
        val_dataset,
        **loader_params,
    )

    path = params.glob_params["prepared_data_localize_path"]
    dataset_utils.save_dataloader(train_loader, path, "train.pt")
    dataset_utils.save_dataloader(val_loader, path, "val.pt")

    minibatch = next(iter(train_loader))[:2]
    dataset_utils.print_batch_features(
        # we only need the image and the bb
        minibatch,
        labels=("Plate image", "Bounding Box"),
    )

    dataset_utils.save_samples(
        minibatch, params.glob_params["out_prepared_localize_path"]
    )


if __name__ == "__main__":
    main()
