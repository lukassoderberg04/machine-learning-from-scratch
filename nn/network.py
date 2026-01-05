from __future__ import annotations

from nn.layer import Layer
from nn.cost import Cost
from mnist.mnist_dataloader import MnistDataloader
import numpy as np

class Network():
    """
        A base class for defining a neural network. Default behaviour is sequential.
    """

    def __init__(self: "Network") -> None:
        self._layers: list[Layer] | None = None
        self._cost: Cost | None          = None
        self._learningRate: float | None = None

    def TrainOneEpoch(self, dataloader: MnistDataloader) -> float:
        """
            Goes through all the batches defined by the dataloader
            and trains the model.

            :param dataloader: The one responsible for loading the training data.
            :type dataloader: mnist.MnistDataloader

            :return: The average cost for this epoch.
            :rtype: float
        """
        if self._cost is None:
            raise RuntimeError("The cost function has not been defined yet!")
        
        if self._layers is None or len(self._layers) <= 0:
            raise RuntimeError("The layers are either undefined or there aren't any layers!")
        
        if self._learningRate is None or self._learningRate < 0:
            raise RuntimeError("The training rate is not defined or below 0!")
        
        avgCost: float = 0
        batches: int   = 0

        while True:
            batch: list[MnistDataloader.DataPair] = dataloader.ReadOneBatch()

            if len(batch) <= 0: return avgCost / batches if batches > 0 else 0.0 # No more pairs to read.

            batches += 1

            cost: float = self._trainOneBatch(batch)

            avgCost += cost

    def _trainOneBatch(self, batch: list[MnistDataloader.DataPair]) -> float:
        """
            Trains one batch!

            :param batch: The batch.
            :type batch: list[mnist.MnistDataloader.DataPair]

            :return: Average cost for this batch.
            :rtype: float
        """
        avgCost: float = 0.0

        for (classification, image) in batch:
            output: np.ndarray = self._forward(image)

            expected: np.ndarray     = np.zeros(shape=10)
            expected[classification] = 1.0 # Ex: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0].

            (derivatives, cost) = self._cost.ComputeCost(output, expected, self._learningRate)

            avgCost += cost

            self._backward(derivatives)

        for layer in self._layers:
            layer.Update(len(batch))

        return avgCost / len(batch)

        
    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        """
            Forwards all the layers and returns the output of the last layer.

            :param inputs: The inputs to the network.
            :type inputs: numpy.ndarray

            :return: The last layer's output.
            :rtype: numpy.ndarray
        """
        lastOutput: np.ndarray = inputs

        for layer in self._layers:
            output: np.ndarray = layer.Forward(lastOutput)
            lastOutput         = output

        return output
    
    def _backward(self, derivatives: np.ndarray) -> None:
        """
            Performs the backward pass for all layers. Only used when training.

            :param derivatives: The derivatives fetched from the cost function.
            :type derivatives: numpy.ndarray
        """
        lastDerivatives: np.ndarray = derivatives

        for layer in reversed(self._layers):
            derivatives: np.ndarray = layer.Backward(lastDerivatives)
            lastDerivatives         = derivatives
    
    def Compute(self, inputs: np.ndarray) -> np.ndarray:
        """
            Computes the model and returns the output.

            :param inputs: The inputs to evaluate.
            :type inputs: numpy.ndarray

            :return: The output of the model.
            :rtype: numpy.ndarray
        """        
        if self._layers is None or len(self._layers) <= 0:
            raise RuntimeError("The layers are either undefined or there aren't any layers!")
        
        outputs: np.ndarray = self._forward(inputs)

        return outputs
    
    def Evaluate(self, dataloader: MnistDataloader) -> float:
        """
            Evaluates the model and return what accuracy it has, from 0 to 1.

            :param dataloader: The one responsible for loading the evaluation data.
            :type dataloader: mnist.MnistDataloader

            :return: The accuracy of this model.
            :rtype: float
        """

        if self._layers is None or len(self._layers) <= 0:
            raise RuntimeError("The layers are either undefined or there aren't any layers!")

        correct: int = 0
        total: int   = 0

        while True:
            batch: list[MnistDataloader.DataPair] = dataloader.ReadOneBatch()

            if len(batch) <= 0:
                break  # no more data to read.

            for (label, image) in batch:
                output: np.ndarray = self.Compute(image)

                predicted: int = int(np.argmax(output))

                if predicted == label:
                    correct += 1

                total += 1

        return correct / total if total > 0 else 0.0

    def CheckLayerConnection(self) -> None:
        """
            Checks that the layer is fully connected, else throws an error!
        """
        if self._layers is None:
            raise RuntimeError("Layers of a neural network has not been initialized yet!")
        
        if len(self._layers) <= 0:
            raise RuntimeError("A network can't have 0 layers!")
        
        for (index, layer) in enumerate(self._layers[:-1]):
            layer_output_size: int = layer.GetOutputSize()

            next_layer: Layer = self._layers[index + 1]
            next_layer_input_size: int = next_layer.GetInputSize()

            if layer_output_size != next_layer_input_size:
                raise RuntimeError(f"Layers don't fully connect! Between layers: {index} and {index + 1}!")