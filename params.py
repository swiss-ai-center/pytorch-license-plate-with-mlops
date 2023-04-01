import os
import typing

import yaml
from dotenv import load_dotenv

from src.generators.swiss_license_plates_generator import Canton

load_dotenv(".env.local")


# -- DVC params --------------------------------------------------------------
BATCH_SIZE = 32


class LocalizeParams:
    # Note: only square images are supported
    IMG_SHAPE = (1, 84, 84)


class PrepareLocalizeParams:
    SEED = 25_03_2023
    MAX_IMAGES = 50_000
    TRAIN_SPLIT = 0.8


class LocalizeModelParams:
    MODEL_NAME = "plate_localizer_v3"
    CONV_LAYERS = (64, "M", 128, "M", 256, "M", 512, "M")
    AVGPOOL_SIZE = (5, 5)
    HIDDEN_LAYERS = (512, 512)
    DROPOUT = 0.25


class TrainLocalizeParams:
    SEED = 25_03_2023
    LR = 0.001
    EPOCHS = 5


class EvaluateLocalizeParams:
    SEED = 25_03_2023


class OCRParams:
    # Note: only square images are supported
    IMG_SHAPE = (1, 84, 84)
    MAX_LABEL_LENGTH = 6
    # number of digits + 1 for the blank symbol
    GRU_NUM_CLASSES = 11
    GRU_BLANK_CLASS = 10
    # number of swiss cantons
    RNN_NUM_CLASSES = len(Canton)  # not for dvc.yaml


class PrepareOCRParams:
    SEED = 25_03_2023
    MAX_IMAGES = 50_000
    TRAIN_SPLIT = 0.8
    # the size of the image before cropping
    IMG_SIZE = (256, 256)


class OCRModelParams:
    MODEL_NAME = "plate_ocr_v2"
    CONV_LAYERS = (64, "M", 128, "M", 256, "M", 256, "M")

    GRU_AVGPOOL_SIZE = (14, 14)
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    GRU_DROPOUT = 0.25

    RNN_AVGPOOL_SIZE = (5, 5)
    RNN_HIDDEN_LAYERS = (1024, 512)
    RNN_DROPOUT = 0.2



class TrainOCRParams:
    SEED = 25_03_2023
    LR = 0.0001
    EPOCHS = 10


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
    PrepareLocalizeParams.MAX_IMAGES = 64
    PrepareOCRParams.MAX_IMAGES = 64
    PrepareStackParams.MAX_IMAGES = 64
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


# -- Python sepecific classes ------------------------------------------------
GlobParamsKeysType = typing.Literal[
    "src_path",
    "out_path",
    "out_prepared_folder",
    "out_log_folder",
    "out_save_folder",
    "out_checkpoint_folder",
    "out_evaluation_folder",
    "localize_model_folder",
    "ocr_model_folder",
    "stack_model_folder",
    "dataset_path",
    "prepared_data_path",
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

err_msg = "params.yaml does not match with params.py"
assert glob_params["localize_model_folder"] == "localize", err_msg
assert glob_params["ocr_model_folder"] == "ocr", err_msg
assert glob_params["stack_model_folder"] == "stack", err_msg

ModelType = typing.Literal["localize", "ocr", "stack"]


class Glob:
    _OUT_PREPARED_PATH = os.path.join(
        glob_params["out_path"], glob_params["out_prepared_folder"]
    )
    _OUT_LOG_PATH = os.path.join(
        glob_params["out_path"], glob_params["out_log_folder"]
    )
    _OUT_SAVE_PATH = os.path.join(
        glob_params["out_path"], glob_params["out_save_folder"]
    )
    _OUT_CHECKPOINT_PATH = os.path.join(
        glob_params["out_path"], glob_params["out_checkpoint_folder"]
    )
    _OUT_EVALUATION_PATH = os.path.join(
        glob_params["out_path"], glob_params["out_evaluation_folder"]
    )

    @staticmethod
    def get(key: GlobParamsKeysType) -> str:
        return glob_params[key]

    @staticmethod
    def get_prepared_data_path(model: ModelType) -> str:
        return os.path.join(glob_params["prepared_data_path"], model)

    @staticmethod
    def get_out_prepared_path(model: ModelType) -> str:
        return os.path.join(Glob._OUT_PREPARED_PATH, model)

    @staticmethod
    def get_out_log_path(model: ModelType) -> str:
        return os.path.join(Glob._OUT_LOG_PATH, model)

    @staticmethod
    def get_out_save_path(model: ModelType) -> str:
        return os.path.join(Glob._OUT_SAVE_PATH, model)

    @staticmethod
    def get_out_checkpoint_path(model: ModelType) -> str:
        return os.path.join(Glob._OUT_CHECKPOINT_PATH, model)

    @staticmethod
    def get_out_evaluation_path(model: ModelType) -> str:
        return os.path.join(Glob._OUT_EVALUATION_PATH, model)
