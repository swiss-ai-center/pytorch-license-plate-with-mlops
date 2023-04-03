import os
import typing

import yaml
from dotenv import load_dotenv

load_dotenv(".env.local")


# -- DVC params --------------------------------------------------------------
BATCH_SIZE = 32


class LocalizeParams:
    # Note: only square images are supported
    IMG_SHAPE = (1, 84, 84)


class PrepareLocalizeParams:
    SEED = 25_03_2023
    MAX_IMAGES = 100_000
    TRAIN_SPLIT = 0.8


class LocalizeModelParams:
    MODEL_NAME = "plate_localizer_v3"
    CONV_LAYERS = (64, "M", 128, "M", 256, "M", 512, "M")
    AVGPOOL_SIZE = (5, 5)
    HIDDEN_LAYERS = (512, 512)
    DROPOUT = 0.4


class TrainLocalizeParams:
    SEED = 25_03_2023
    LR = 0.001
    EPOCHS = 3


class EvaluateLocalizeParams:
    SEED = 25_03_2023


class OCRParams:
    # Note: only square images are supported
    IMG_SHAPE = (1, 84, 84)
    MAX_LABEL_LENGTH = 7
    # cantons (26) + number of digits (10) + 1 for the blank symbol
    GRU_NUM_CLASSES = 37
    GRU_BLANK_CLASS = 36
    # number of swiss cantons
    # RNN_NUM_CLASSES = len(Canton)  # not for dvc.yaml


class PrepareOCRParams:
    SEED = 25_03_2023
    MAX_IMAGES = 250_000
    TRAIN_SPLIT = 0.8
    # the size of the image before cropping
    IMG_SIZE = (256, 256)


class OCRModelParams:
    MODEL_NAME = "plate_ocr_v2"
    CONV_LAYERS = (64, "M", 128, "M", 256, "M", 256, "M")

    GRU_AVGPOOL_SIZE = (18, 18)
    GRU_HIDDEN_SIZE = 384
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.5


class TrainOCRParams:
    SEED = 25_03_2023
    LR = 0.00025
    EPOCHS = 2


class EvaluateOCRParams:
    SEED = 25_03_2023


class PrepareStackParams:
    SEED = 25_03_2023
    MAX_IMAGES = 5_000


class EvaluateStackParams:
    SEED = 25_03_2023


IS_LOCAL = bool(int(os.environ.get("IS_LOCAL", False)))
GLOB_SEED = os.environ.get("GLOB_SEED", None)

if IS_LOCAL:
    # MAX_IMAGES should respect MAX_IMAGES * TRAIN_SPLIT > BATCH_SIZE as the
    # last batch will be dropped if it is smaller than BATCH_SIZE
    PrepareLocalizeParams.MAX_IMAGES = 256
    PrepareOCRParams.MAX_IMAGES = 256
    PrepareStackParams.MAX_IMAGES = 256
    TrainLocalizeParams.EPOCHS = 1
    TrainOCRParams.EPOCHS = 1


if GLOB_SEED is not None:
    PrepareLocalizeParams.SEED = int(GLOB_SEED)
    TrainLocalizeParams.SEED = int(GLOB_SEED)
    EvaluateLocalizeParams.SEED = int(GLOB_SEED)
    PrepareOCRParams.SEED = int(GLOB_SEED)
    TrainOCRParams.SEED = int(GLOB_SEED)
    EvaluateOCRParams.SEED = int(GLOB_SEED)
    PrepareStackParams.SEED = int(GLOB_SEED)
    EvaluateStackParams.SEED = int(GLOB_SEED)


# -- Python specific  --------------------------------------------------------
# The following are typing extensions to make catching errors easier
GlobParamsKeysType = typing.Literal[
    "dataset_path",
    "prepared_data_localize_path",
    "prepared_data_ocr_path",
    "prepared_data_stack_path",
    "out_prepared_localize_path",
    "out_prepared_ocr_path",
    "out_log_localize_path",
    "out_log_ocr_path",
    "out_save_localize_path",
    "out_save_ocr_path",
    "out_checkpoints_localize_path",
    "out_checkpoints_ocr_path",
    "out_evaluation_localize_path",
    "out_evaluation_ocr_path",
    "out_evaluation_stack_path",
]

GlobParamsType = typing.Dict[
    GlobParamsKeysType,
    str,
]

glob_params: GlobParamsType = yaml.safe_load(open("params.yaml"))

# Verify that the params.yaml and params.py are in sync
err_msg = "params.yaml keys does not match with params.py keys"
assert (
    tuple(glob_params.keys()) == GlobParamsType.__args__[0].__args__
), err_msg
