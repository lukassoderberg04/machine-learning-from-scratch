from __future__ import annotations

from nn.layer import Layer
import numpy as np

class Softmax(Layer):
    """
        A softmax layer is a layer which is used in the last output layer
        of a neural network to turn logits (which are computed scores) into
        probabilities that all sum up to 1. This is very useful for models
        which do multi-class classification. For example: classifying an image
        by being either a dog, cat or bird (3 classes).
    """

    def __init__(self: "Softmax", inputs: int):
        """
            :param inputs: The amount of inputs.
            :type inputs: int
        """

        self._size: tuple[int, int] = (inputs, inputs) # The same size since it just applies a function on the input.

        # Initialized to None since no forward pass has happened.
        self._inputs: np.array | None  = None
        self._outputs: np.array | None = None

    def Forward(self, inputs: np.array) -> np.array:
        """
            The softmax has the function: e^(input - c) / sum(inputs, e^(input - c)), c = max(inputs). The exponential makes sure
            that the value, even if it's negative, is still positive. The (input - c) part is for making the values numerically stable.
            This is because exponentials get high values very very fast which will cause the output to sometime drift to infinity, hence
            we remove the maximum value found among the inputs with the input. An example: inputs = [4.0, -3.0, 2.0], will become after
            (input - max(inputs)) => [0.0, -7.0, -2.0], since max(inputs) = 4.0. The dividing part is just taking the sum of all new
            "scores" calculated and dividing each score by that sum. Just like you would when learning about probability when you were small.
            Ex: 5 apples, 5 pears, how many percent apples. Well, you take 5 / (5 + 5) = 0.5.

            :param inputs: The logits to be converted into probabilities.
            :type inputs: numpy.array

            :return: The probabilities for each class.
            :rtype: numpy.array
        """
        if len(inputs) != self.GetInputSize():
            raise RuntimeError("The inputs size didn't match the softmax's layer size!")
        
        # Set the value for history.
        self._inputs = inputs
        
        # Finds the maximum value of the inputs.
        max: float = np.max(inputs)

        # Elementwise remove max from the original value, hence making it numerically stable.
        shifted: np.array = inputs - max

        # Elementwise take e^(input).
        exponentialInputs: np.array = np.exp(shifted)

        # Elementwise take the computed exponential and divide by the sum of all exponentials.
        self._outputs = exponentialInputs / np.sum(exponentialInputs)

        return self._outputs
    
    def Backward(self, derivatives: np.array) -> np.array:
        pass