from typing import Tuple, Union

import torch
from torch import Tensor, nn

import params
from src.models.abstract_model import AbstractModel


class CNNOCRModel(AbstractModel):
    """CNN model for object multi-classification."""

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        conv_layers: Tuple[Union[int, str]],
        gru_avgpool_size: Tuple[int, int],
        gru_hidden_size: int,
        gru_num_layers: int,
        gru_num_classes: int,
        gru_dropout: float,
        rnn_avgpool_size: Tuple[int, int],
        rnn_hidden_layers: Tuple[int],
        rnn_num_classes: int,
        rnn_dropout: float,
    ):
        """
        Initialize the model.

        Args:
            img_shape (Tuple[int, int, int]): Image shape
                (channels, height, width)
            conv_layers (Tuple[int | str]): List of convolutional layers. Each
                element can be an integer (number of features) or a string
                ("M" for maxpooling).
            gru_avgpool_size (Tuple[int, int]): Size of the average pooling layer.
                It should be a tuple of two integers.
            gru_hidden_size (int): Number of features in the hidden state of
                the GRU.
            gru_num_layers (int): Number of recurrent layers.
            gru_num_classes (int): Number of classes for the text/digit
                recognition including the blank class. (c.f. CTC loss)
            gru_dropout (float): Dropout rate for the GRU layer.
            rnn_avgpool_size (Tuple[int, int]): Size of the average pooling layer.
                It should be a tuple of two integers.
            rnn_hidden_layers (Tuple[int]): List of hidden layers for the
                RNN caterigorical classification.
            rnn_num_classes (int): Number of classes for an additional
                classification layer.
            rnn_dropout (float): Dropout rate for the RNN.
        """
        super(CNNOCRModel, self).__init__()

        self.features_layers_conv = []
        in_channels = img_shape[0]
        for layer_features in conv_layers:
            if layer_features == "M":
                self.features_layers_conv += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            else:
                self.features_layers_conv += [
                    nn.Conv2d(
                        in_channels, layer_features, kernel_size=3, padding=1
                    ),
                    nn.ReLU(inplace=True),
                ]
                in_channels = layer_features

        self.features = nn.Sequential(*self.features_layers_conv)
        self.gru_avgpool = nn.AdaptiveAvgPool2d(gru_avgpool_size)
        self.rnn_avgpool = nn.AdaptiveAvgPool2d(rnn_avgpool_size)

        # GRU layer
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.gru_input_size = (
            self.features_layers_conv[-3].out_channels
            * self.gru_avgpool.output_size[0]
        )
        self.gru_layer = nn.GRU(
            input_size=self.gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=True,
        )

        # Add the fully connected layer
        self.out_fc_gru = nn.Linear(gru_hidden_size * 2, gru_num_classes)
        self.log_softmax_out = nn.LogSoftmax(dim=-1)
        self.loss_fn_ctc = nn.CTCLoss(
            blank=params.OCRParams.GRU_BLANK_CLASS,
            reduction="mean",
            zero_infinity=True,
        )

        # RNN layers
        self.flatten = nn.Flatten(start_dim=1)
        self.hidden_rnn = nn.Sequential()
        last_layer_features_rnn = (
            self.features_layers_conv[-3].out_channels
            * self.rnn_avgpool.output_size[0]
            * self.rnn_avgpool.output_size[1]
        )
        for i, layer_features in enumerate(rnn_hidden_layers):
            self.hidden_rnn.add_module(
                str(i),
                nn.Linear(last_layer_features_rnn, layer_features),
            )
            self.hidden_rnn.add_module(str(i + 1), nn.ReLU(inplace=True))
            self.hidden_rnn.add_module(
                str(i + 2), nn.Dropout(rnn_dropout, inplace=False)
            )

            last_layer_features_rnn = layer_features

        self.out_fc_rnn = nn.Linear(last_layer_features_rnn, rnn_num_classes)
        self.softmax_out = nn.Softmax(dim=-1)
        self.loss_fn_ce = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        # CNN layers
        x = self.features(x)

        # RNN addtionnal classification
        x_rnn = self.rnn_avgpool(x)
        x_rnn = self.flatten(x_rnn)
        x_rnn = self.hidden_rnn(x_rnn)
        x_rnn = self.out_fc_rnn(x_rnn)
        x_rnn = self.softmax_out(x_rnn)

        # GRU text/digit recognition
        x_gru = self.gru_avgpool(x)
        x_gru = x_gru.clone()
        x_gru = x_gru.permute(0, 3, 2, 1)
        x_gru = x_gru.reshape(x_gru.shape[0], -1, self.gru_input_size)
        x_gru, _ = self.gru_layer(x_gru)
        x_gru = torch.stack(
            [
                self.log_softmax_out(self.out_fc_gru(x_gru[i]))
                for i in range(x_gru.shape[0])
            ]
        )
        return (x_gru, x_rnn)

    def loss(self, y_pred: Tuple[Tensor, Tensor], y_true: Tensor) -> Tensor:
        """Compute the loss."""
        # Permute the dimensions to have the input length as the first
        # dimension
        pred_gru, pred_rnn = y_pred
        pred_gru = pred_gru.permute(1, 0, 2)

        true_gru, true_rnn = (
            y_true[:, : params.OCRParams.MAX_LABEL_LENGTH],
            # The last column is the class index
            y_true[:, -1],
        )
        # Compute the loss for the text/digit recognition
        input_lengths = torch.full(
            size=(pred_gru.shape[1],), fill_value=pred_gru.shape[0]
        )
        # for each target (true_gru) in the batch, set target length to its
        # length
        target_lengths = torch.IntTensor([len(target) for target in true_gru])

        loss_num = self.loss_fn_ctc(
            pred_gru,
            true_gru,
            input_lengths,
            target_lengths,
        )
        # Compute the loss for the additional classification
        loss_canton = self.loss_fn_ce(pred_rnn, true_rnn).item()

        return loss_num + loss_canton
