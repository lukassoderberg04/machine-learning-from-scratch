from __future__ import annotations

import numpy as np

class Layer():
    """
        A base class for defining specific layers for neural networks.
        Inputs flow through the system and compute the output depending on
        some function defined by the class that implements this class.
    """

    def __init__(self: "Layer") -> None:
        self._size: tuple[int, int] | None = None # Size of the input and output array as a tuple.

    def Forward(self, inputs: np.array) -> np.array:
        """
            Computes the output based on inputs.

            :param inputs: Input neurons for this layer.
            :type inputs: numpy.array

            :return: The computed output.
            :rtype: numpy.array
        """
        raise NotImplementedError("Forward has not been implemented by the class yet!")

    def Backward(self, derivatives: np.array) -> np.array:
        """
            Computes the local derivative and pushes the derivative back in the chain.

            :param derivatives: The computed derivatives from the layer in front.
            :type derivatives: numpy.array

            :return: The computed last derivative which is to be used by the layer behind this. 
            :rtype: numpy.array
        """
        raise NotImplementedError("Backward has not been implemented by the class yet!")
    
    def GetSize(self) -> tuple[int, int]:
        if self._size is None:
            raise RuntimeError("Size has not been initialized by a layer!")
        
        return self._size
    
    def GetInputSize(self) -> int:
        return self.GetSize()[0]
    
    def GetOutputSize(self) -> int:
        return self.GetSize()[1]