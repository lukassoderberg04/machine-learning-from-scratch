from __future__ import annotations

import numpy as np

class Cost:
    """
        The cost class is a base class for all cost functions. It's purpose is to
        evaluate how bad the model did for a specific test subject.
    """

    def __init__(self: "Cost"):
        """
            :param inputs: Total amount of inputs going into this cost function.
        """
        self._size: int | None = None

    def ComputeCost(self, inputs: np.ndarray, expected: np.ndarray, learningRate: float) -> tuple[np.ndarray, float]:
        """
            Should be implemented by another class and should compute the total cost
            as well as the derivative.

            :param inputs: The input to be evaluated against the expected. The length of this
            should be defined in the constructor.
            :type inputs: numpy.ndarray

            :param expected: The expected input.
            :type expected: numpy.ndarray

            :param learningRate: The scalar of the cost.
            :type learningRate: float

            :return: The local derivatives in terms of the input together with the total cost: (derivatives, cost).
            :rtype: tuple[numpy.ndarray, float]
        """

        raise NotImplementedError("Can't use the Cost class own ComputeCost!")