from torch import Tensor, nn
from torchvision import models


class VGG16Model(nn.Module):
    def __init__(
        self,
        img_shape: list[int, int, int],
        hidden_layers: list[int],
        fc_features_in: int,
        num_classes: int,
        dropout: float,
    ):
        super(VGG16Model, self).__init__()
        # Set input size
        # Add the hidden layers
        self.hidden_layers_linear = []
        last_layer_features = 1000
        for i, layer_features in enumerate(hidden_layers):
            out_features = layer_features
            if i == len(hidden_layers) - 1:
                out_features = fc_features_in
            self.hidden_layers_linear.extend(
                [
                    nn.Linear(last_layer_features, out_features),
                    nn.ReLU(inplace=True),
                ]
            )
            last_layer_features = out_features

        # Add the dropout layer
        self.hidden_layers_linear.append(nn.Dropout(dropout))

        self.pretrained_model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT,
        )

        # Freeze all the parameters in the network
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        conv1 = self.pretrained_model.features[0]
        if img_shape[0] != conv1.in_channels:
            self.pretrained_model.features[0] = nn.Conv2d(
                img_shape[0],
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )

        self.hiddens = nn.Sequential(*self.hidden_layers_linear)
        # Replace the first layer with a new one
        # conv1 = self.pretrained_model.conv1
        # self.pretrained_model.conv1 = nn.Conv2d(
        #     img_shape[0],
        #     conv1.out_channels,
        #     kernel_size=conv1.kernel_size,
        #     stride=conv1.stride,
        #     padding=conv1.padding,
        #     bias=False,
        # )

        # Replace the fully connected layer with a new one
        self.out_fc = nn.Linear(fc_features_in, num_classes)

    def forward(self, x: Tensor):
        x = self.pretrained_model(x)
        x = self.hiddens(x)
        x = self.out_fc(x)
        return x

    def to(self, *args, **kwargs):
        model = super().to(*args, **kwargs)
        self.hiddens.to(*args, **kwargs)
        self.pretrained_model.to(*args, **kwargs)
        return model
