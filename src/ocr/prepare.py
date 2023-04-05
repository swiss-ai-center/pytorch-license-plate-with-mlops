from torchvision import transforms

import params
import src.utils.dataset as dataset_utils
from src.datasets.swiss_license_plates_ocr_dataset import (
    SwissLicensePlateOCRDataset,
)
from src.utils.seed import set_seed


def main():
    set_seed(params.PrepareOCRParams.SEED)

    img_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
        ]
    )

    dataset = SwissLicensePlateOCRDataset(
        img_size_before_crop=params.PrepareOCRParams.IMG_SIZE,
        template_path=params.glob_params["dataset_path"],
        img_shape=params.OCRParams.IMG_SHAPE,
        img_transform=img_transform,
        max_images=params.PrepareOCRParams.MAX_IMAGES,
    )

    train_dataset, val_dataset = dataset_utils.split_dataset(
        dataset, params.PrepareOCRParams.TRAIN_SPLIT
    )

    loader_params = {
        "multiprocessing": params.MULTIPROCESSING,
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

    path = params.glob_params["prepared_data_ocr_path"]
    dataset_utils.save_dataloader(train_loader, path, "train.pt")
    dataset_utils.save_dataloader(val_loader, path, "val.pt")

    minibatch = next(iter(train_loader))
    dataset_utils.print_batch_features(
        minibatch,
        labels=("Cropped plate image", "Hot encoded plate data"),
    )

    dataset_utils.save_samples(
        minibatch, params.glob_params["out_prepared_ocr_path"]
    )


if __name__ == "__main__":
    main()
