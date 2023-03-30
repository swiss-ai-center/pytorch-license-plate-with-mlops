from torchvision import transforms

import params
import src.utils.dataset as dataset_utils
from src.datasets.swiss_license_plates_dataset import SwissLicensePlatesDataset
from src.utils.seed import set_seed


def main():
    set_seed(params.PrepareStackParams.SEED)

    img_transform = transforms.Compose(
        [
            transforms.ColorJitter(brightness=0.5, hue=0.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
            transforms.RandomAutocontrast(),
            transforms.ToTensor(),
        ]
    )

    dataset = SwissLicensePlatesDataset(
        template_path=params.Glob.get("dataset_path"),
        img_shape=(
            params.LocalizeParams.IMG_SHAPE[0],
            *params.PrepareOCRParams.IMG_SIZE,
        ),
        img_transform=img_transform,
        max_images=params.PrepareStackParams.MAX_IMAGES,
    )

    val_loader = dataset_utils.get_data_loader(
        dataset,
        multiprocessing=(not params.IS_LOCAL),
        batch_size=params.BATCH_SIZE,
    )

    path = params.Glob.get_prepared_data_path("stack")
    dataset_utils.save_dataloader(val_loader, path, "val.pt")


if __name__ == "__main__":
    main()
