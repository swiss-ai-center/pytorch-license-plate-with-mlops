from abc import ABCMeta, abstractmethod

from torch import Tensor, nn


class AbstractModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        pass
