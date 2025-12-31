from __future__ import annotations

from layer import Layer
import numpy as np

class DenseLayer(Layer):
    """
        A layer in which all input nodes, connect to all output nodes, where all
        connections has a weight attached to it and all output nodes a bias.
    """

    def __init__(self: "Layer", inputCounts: int, outputCounts: int, learningRate: float) -> None:
        self._input:  np.array | None      = np.zeros(shape=inputCounts)
        self._output: np.array | None      = np.zeros(shape=outputCounts)
        self._size: tuple[int, int] | None = (inputCounts, outputCounts)
        self._learningRate: float | None   = learningRate