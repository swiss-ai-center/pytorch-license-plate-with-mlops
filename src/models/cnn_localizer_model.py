from typing import Tuple, Union

from torch import Tensor, nn

from src.models.abstract_model import AbstractModel


class CNNLocaliserModel(AbstractModel):
    """CNN model for object localization."""

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        conv_layers: Tuple[Union[int, str]],
        avgpool_size: Tuple[int, int],
        hidden_layers: Tuple[int],
        dropout: float,
    ):
        """
        Initialize the model.

        Args:
            img_shape (Tuple[int, int, int]): Image shape
                (channels, height, width)
            conv_layers (Tuple[int | str]): List of convolutional layers. Each
                element can be an integer (number of features) or a string
                ("M" for maxpooling).
            avgpool_size (Tuple[int, int]): Size of the average pooling layer.
                It should be a tuple of two integers.
            hidden_layers (Tuple[int]): List of hidden layers where each
                element is the number of features.
            dropout (float): Dropout rate.
        """
        super(CNNLocaliserModel, self).__init__()

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
        self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
        self.flatten = nn.Flatten(start_dim=1)

        # Add the hidden layers
        self.hidden_layers_linear = []
        # Here we initialize the first layer with the input size of the
        # avgpool output
        last_layer_features = (
            self.features_layers_conv[-3].out_channels
            * self.avgpool.output_size[0]
            * self.avgpool.output_size[1]
        )
        for layer_features in hidden_layers:
            self.hidden_layers_linear.extend(
                [
                    nn.Linear(last_layer_features, layer_features),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout, inplace=False),
                ]
            )
            last_layer_features = layer_features
        self.hiddens = nn.Sequential(*self.hidden_layers_linear)

        # Add the fully connected layer with a new one with 4 outputs for the
        # bounding box
        self.out_fc = nn.Linear(hidden_layers[-1], 4)
        self.sig_out = nn.Sigmoid()
        self.loss_fn = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # CNN layers
        x = self.features(x)
        x = self.avgpool(x)

        # FC layers
        x = self.flatten(x)
        x = self.hiddens(x)
        x = self.out_fc(x)
        x = self.sig_out(x)
        return x

    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the loss."""
        return self.loss_fn(y_pred, y_true)
