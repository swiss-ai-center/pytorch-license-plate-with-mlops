from torch import Tensor, nn
from torchvision import models


class VGG16Model(nn.Module):
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        hidden_layers: tuple[int],
        num_classes: int,
        dropout: float,
    ):
        super(VGG16Model, self).__init__()

        vgg16 = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT,
        )

        # Freeze all the parameters in the network
        for param in vgg16.parameters():
            param.requires_grad = False

        # Replace the first layer with a new one to fit image channels
        conv1 = vgg16.features[0]
        if img_shape[0] != conv1.in_channels:
            vgg16.features[0] = nn.Conv2d(
                img_shape[0],
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=False,
            )

        self.features = nn.Sequential(*vgg16.features)
        self.avgpool = vgg16.avgpool
        self.flatten = nn.Flatten(start_dim=1)

        # Add the hidden layers
        self.hidden_layers_linear = []
        # Here we initialize the first layer with the input size of the
        # avgpool output (512 * 7 * 7) in this case
        last_layer_features = vgg16.classifier[0].in_features
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

        # Add the fully connected layer with a new one
        self.out_fc = nn.Linear(hidden_layers[-1], num_classes)
        self.sig_out = nn.Sigmoid()

    def forward(self, x: Tensor):
        # Pretrained model
        x = self.features(x)
        x = self.avgpool(x)
        # FC layers
        x = self.flatten(x)
        x = self.hiddens(x)
        x = self.out_fc(x)
        x = self.sig_out(x)
        return x
