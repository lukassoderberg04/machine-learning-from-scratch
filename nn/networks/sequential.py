from __future__ import annotations

from nn.network import Network
from nn.layer import Layer
from nn.cost import Cost
from mnist.mnist_dataloader import MnistDataloader

class Sequential(Network):
    """
        A sequential network executes layers in sequence.
    """

    def __init__(self: "Sequential", layers: list[Layer], cost: Cost, learningRate: float = 0.01) -> None:
        """
            :param layers: A list of layers, computed in sequence of each other.
            :type layers: list[nn.Layer]

            :param cost: The cost object to take care of calculating the error.
            :type cost: nn.Cost

            :param learningRate: The rate at which learning will occur. Lower values often mean more stable
            but longer training time.
            :type learningRate: float
        """
        self._layers: list[Layer] = layers
        self._cost: Cost          = cost
        self._learningRate: float = learningRate