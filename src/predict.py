import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from models.vgg16_model import VGG16Model
from utils.preview_utils import draw_bb

PATH = "models/checkpoints_20230324_004557/model_16"
glob_params = yaml.safe_load(open("params.yaml"))

model = VGG16Model(
    glob_params["img_shape"],
    glob_params["hidden_layers"],
    glob_params["fc_features_in"],
    glob_params["num_classes"],
    glob_params["dropout"],
)
model.load_state_dict(torch.load(PATH))

val_loader = torch.load(os.path.join("data", "prepared", "val.pt"))


for i, vdata in enumerate(val_loader):
    vinputs, vlabels = vdata
    voutputs = model(vinputs)

    # show the first 32 images with matplotlib
    # TODO: show this only locally

    plt.figure(figsize=(16, 8))
    for i in range(glob_params["batch_size"]):
        img = (
            (vinputs[i].numpy() * 255)
            .astype(np.uint8)
            .transpose((1, 2, 0))
            .copy()
        )
        bb = vlabels[i].numpy() * img.shape[1]
        bb_pred = voutputs[i].detach().numpy() * img.shape[1]
        img = draw_bb(img, bb)
        img = draw_bb(img, bb_pred, color=(255, 0, 0))
        plt.subplot(4, 8, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
