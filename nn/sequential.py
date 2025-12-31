from __future__ import annotations

from layer import Layer
from mnist.mnist_dataloader import MnistDataloader
from mnist.mnist_image import MnistImage

class Sequential():
    """
        A sequential network that execute layer in order and use the
        output from the previous layer as input in the next.
    """
    Cost = float # The cost type.
    
    def __init__(self: "Sequential", layers: list[Layer], learningRate: float = 0.01) -> None:
        """
            :param layers: The layers of the neural network.
            :type layers: list[Layer]
            :param learningRate: Determins the gradient's influence when training.
            :type learningRate: float
        """
        if (len(layers) < 1):
            raise RuntimeError("Can't define a network with zero layers!")
        
        if (learningRate == 0):
            raise RuntimeError("The learning rate can't be 0 since the model won't learn anything!")

        self._layers: list[Layer] = layers
        self._learningRate: float = learningRate

    def TrainOneEpoch(self, dataloader: MnistDataloader) -> None:
        """
            Trains through the whole dataset once.
                
            :param dataloader: Loads the training data.
            :type dataloader: MnistDataloader
        """
        pairs: list[MnistDataloader.DataPair] = dataloader.ReadOneBatch()
        
        while (len(pairs) > 0):
            self._trainWithData(pairs)

            pairs = dataloader.ReadOneBatch()

    def _trainWithData(self, data: list[MnistDataloader.DataPair]) -> None:
        raise NotImplementedError("Private function: _trainWithData, has not been implemented yet!")