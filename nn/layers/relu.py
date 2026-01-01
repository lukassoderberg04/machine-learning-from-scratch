from __future__ import annotations

from nn.layer import Layer
import numpy as np

class Relu(Layer):
    """
        ReLU - Rectified Linear Unit, is an activation function, and in this case an
        activation layer. It's seen as a linear function when x > 0 and flat 0 otherwise.
    """

    def __init__(self: "Relu", inputs: int):
        self._size: tuple[int, int] = (inputs, inputs) # The same size since it just applies a function on the input.

        # Initialized to None since no forward pass has happened.
        self._inputs: np.array | None  = None
        self._outputs: np.array | None = None

    def Forward(self, inputs: np.array) -> np.array:
        """
            The forward of the ReLU function is just a max(0, input), since it's a y(x) = x function when
            x > 0 and else 0.
        """
        if len(inputs) != self.GetInputSize():
            raise RuntimeError("ReLU input for forwarding is not of the correct size as defined by the constructor!")

        self._inputs = inputs

        # Applies the ReLU function on the input and computes the output.
        self._outputs = np.maximum(0, self._inputs)

        return self._outputs
    
    def Backward(self, derivatives: np.array) -> np.array:
        """
            The backward pass on the ReLU function calculates the dactivation / dinput. And since
            the activation function looks like this: activation(input) = input, when input > 0, the
            derivative for input > 0 is going to be 1. And when input <= 0, it will be 0.
        """
        if self._outputs is None:
            raise RuntimeError("ReLU function hasn't yet executed the forward step!")
        
        if len(derivatives) != self.GetOutputSize():
            raise RuntimeError("The derivative size doesn't match the output size of the ReLU function!")
        
        # Since we need to propogate the derivative in relation to the input (since the input is calculated in the layers before),
        # we multiply the derivatives where the input is <= 0 with 0 and else 1. This is done due to the chain rule.
        toPropagate: np.array = derivatives * (self._inputs > 0)
        
        return toPropagate