from __future__ import annotations

import numpy as np
from nn.cost import Cost

class Mse(Cost):
    """
        The MSE - Mean Square Error class evaluates a model by using the mean square error.
    """

    def __init__(self: "Mse", inputs: int):
        """
            :param inputs: Total amount of inputs going into this cost function.
        """
        self._size: int = inputs

    def ComputeCost(self, inputs: np.ndarray, expected: np.ndarray, learningRate: float) -> tuple[np.ndarray, float]:
        """
            The mean square error function: mse(inputs, expected) = (1 / len(inputs)) * sum((inputs - expected) ** 2).

            :param inputs: The inputs of which to be evaluated against the expected inputs.
            :type inputs: numpy.ndarray

            :param expected: The expected inputs.
            :type expected: numpy.ndarray

            :param learningRate: The scalar for the cost.
            :type learningRate: float

            :return: The derivatives and the cost as a tuple: (derivatives, cost).
            :rtype: tuple[numpy.ndarray, float]
        """
        if len(inputs) != self._size:
            raise RuntimeError("The inputs for computing the cost using MSE is not of the correct size!")
        
        if len(expected) != self._size:
            raise RuntimeError("The expected for computing the cost using MSE is not of the correct size!")
        
        if learningRate <= 0:
            raise RuntimeError("Learning rate has to be bigger than 0!")
        
        inputCount: int = len(inputs)

        meanSquareError: float = (1 / inputCount) * np.sum(np.pow((inputs - expected), 2))

        # This is derived by taking dMSE(input) / dinput!
        derivatives: np.ndarray = 2 * (inputs - expected) / inputCount

        return (derivatives, meanSquareError * learningRate)