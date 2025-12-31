from __future__ import annotations

import numpy as np

class Layer():
    """
        A base class for defining specific layers for neural networks.
        Inputs flow through the system and compute the output depending on
        some function defined by the class that implements this class.
    """

    def __init__(self: "Layer") -> None:
        self._input:  np.array | None      = None
        self._output: np.array | None      = None
        self._size: tuple[int, int] | None = None # Size of the input and output array as a tuple.
        self._learningRate: float | None   = None # The change of which to change the inner workings of this layer.

        return
    
    def ForwardWithInput(self, input: np.array) -> None:
        """
            Forwards, but first sets the input!
        
            :param input: Input to the system.
            :type input: numpy.array
        """
        self._setInput(input)
        self.Forward()
    
    def Forward(self) -> None:
        """
            Calculates the output based on the layer's input. Needs to be implemented.
            The type of input will be defined by the layer that implement this base class.

            :param input: The input neurons.
            :type batchSize: numpy.array
        """
        raise NotImplementedError("Forward has not been implemented by the class yet!")

    def Backward(self, derivatives: np.array) -> np.array:
        """
            Computes the local derivative and changes it's inner workings based on that.

            :param derivatives: The computed derivatives from the layer in front.
            :type derivatives: numpy.array
            :return: The computed last derivative which is to be used by the layer after this. 
            :rtype: numpy.array
        """
        raise NotImplementedError("Backward has not been implemented by the class yet!")
    
    def _setLearningRate(self, value: float) -> None:
        self._learningRate = value

    def _getLearningRate(self) -> float:
        if (self._learningRate is None):
            raise RuntimeError("Learning rate has not been set in the layer!")
        
        return self._learningRate

    def _setInput(self, value: np.array) -> None:
        """
            Sets the input using the value.
        """
        if (value is not None and len(value) != len(self._getInput())):
            raise RuntimeError(f"The length of the previous input: {len(self._getInput())} doesn't match the length of value: {len(value)}!")

        self._input = value

    def _getInput(self) -> np.array:
        """
            Getter for the input.
        """
        if (self._input is None):
            raise RuntimeError("Input has not been set yet.")

        return self._input
    
    def GetOutput(self) -> np.array:
        """
            Getter for output.
        """
        if (self._output is None):
            raise RuntimeError("Output has not been calculated yet.")

        return self._output
    
    def GetSize(self) -> tuple[int, int]:
        """
            Returns the size of the layer.
        
            :return: The returned value is formatted like this: (len(_input), len(_output))!
            :rtype: tuple[int, int]
        """
        if (self._size is None):
            raise RuntimeError("Can't get size since size still is not defined!")
        
        return self._size
    
    def _setSize(self, value: tuple[int, int]) -> None:
        self._size = value