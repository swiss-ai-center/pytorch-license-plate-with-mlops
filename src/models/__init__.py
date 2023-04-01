import params
from src.models.cnn_localizer_model import CNNLocaliserModel
from src.models.cnn_ocr_model import CNNOCRModel

model_registry = {
    "plate_localizer_v3": CNNLocaliserModel(
        params.LocalizeParams.IMG_SHAPE,
        params.LocalizeModelParams.CONV_LAYERS,
        params.LocalizeModelParams.AVGPOOL_SIZE,
        params.LocalizeModelParams.HIDDEN_LAYERS,
        params.LocalizeModelParams.DROPOUT,
    ),
    "plate_ocr_v2": CNNOCRModel(
        params.OCRParams.IMG_SHAPE,
        params.OCRModelParams.CONV_LAYERS,
        params.OCRModelParams.GRU_AVGPOOL_SIZE,
        params.OCRModelParams.GRU_HIDDEN_SIZE,
        params.OCRModelParams.GRU_NUM_LAYERS,
        params.OCRParams.GRU_NUM_CLASSES,
        params.OCRModelParams.GRU_DROPOUT,
        params.OCRModelParams.RNN_AVGPOOL_SIZE,
        params.OCRModelParams.RNN_HIDDEN_LAYERS,
        params.OCRParams.RNN_NUM_CLASSES,
        params.OCRModelParams.RNN_DROPOUT,
    ),
}
