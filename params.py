import typing

import yaml

# -- DVC params --------------------------------------------------------------
MULTIPROCESSING = True
BATCH_SIZE = 32


class LocalizeParams:
    # Note: only square images are supported
    IMG_SHAPE = (1, 84, 84)


class PrepareLocalizeParams:
    SEED = 25032023
    MAX_IMAGES = 100000
    TRAIN_SPLIT = 0.8


class LocalizeModelParams:
    MODEL_NAME = "plate_localizer_v3"
    CONV_LAYERS = (64, "M", 128, "M", 256, "M", 512, "M")
    AVGPOOL_SIZE = (5, 5)
    HIDDEN_LAYERS = (512, 512)
    DROPOUT = 0.4


class TrainLocalizeParams:
    SEED = 25032023
    LR = 0.0005
    EPOCHS = 3


class EvaluateLocalizeParams:
    SEED = 25032023


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
    SEED = 25032023
    MAX_IMAGES = 250000
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
    SEED = 25032023
    LR = 0.00025
    EPOCHS = 2


class EvaluateOCRParams:
    SEED = 25032023


class PrepareStackParams:
    SEED = 25032023
    MAX_IMAGES = 5000


class EvaluateStackParams:
    SEED = 25032023


# -- Python specific  --------------------------------------------------------
# The following are typing extensions to make catching errors easier
GlobParamsKeysType = typing.Literal[
    "src_path",
    "out_path",
    "dataset_path",
    "prepared_data_path",
    "out_prepared_folder",
    "out_log_folder",
    "out_checkpoint_folder",
    "out_save_folder",
    "out_evaluation_folder",
    "localize_model_folder",
    "ocr_model_folder",
    "stack_model_folder",
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
