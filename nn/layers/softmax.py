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
        self._inputs: np.ndarray | None  = None
        self._outputs: np.ndarray | None = None

    def Forward(self, inputs: np.ndarray) -> np.ndarray:
        """
            The softmax has the function: e^(input - c) / sum(inputs, e^(input - c)), c = max(inputs). The exponential makes sure
            that the value, even if it's negative, is still positive. The (input - c) part is for making the values numerically stable.
            This is because exponentials get high values very very fast which will cause the output to sometime drift to infinity, hence
            we remove the maximum value found among the inputs with the input. An example: inputs = [4.0, -3.0, 2.0], will become after
            (input - max(inputs)) => [0.0, -7.0, -2.0], since max(inputs) = 4.0. The dividing part is just taking the sum of all new
            "scores" calculated and dividing each score by that sum. Just like you would when learning about probability when you were small.
            Ex: 5 apples, 5 pears, how many percent apples. Well, you take 5 / (5 + 5) = 0.5.

            :param inputs: The logits to be converted into probabilities.
            :type inputs: numpyp.ndarray

            :return: The probabilities for each class.
            :rtype: numpyp.ndarray
        """
        if len(inputs) != self.GetInputSize():
            raise RuntimeError("The inputs size didn't match the softmax's layer size!")
        
        # Set the value for history.
        self._inputs = inputs
        
        # Finds the maximum value of the inputs.
        max: float = np.max(inputs)

        # Elementwise remove max from the original value, hence making it numerically stable.
        shifted: np.ndarray = inputs - max

        # Elementwise take e^(input).
        exponentialInputs: np.ndarray = np.exp(shifted)

        # Elementwise take the computed exponential and divide by the sum of all exponentials.
        self._outputs = exponentialInputs / np.sum(exponentialInputs)

        return self._outputs
    
    def Backward(self, derivatives: np.ndarray) -> np.ndarray:
        """
            First of, this layer isn't that effective since it has to calculate the jacobian matrix
            containing all the different gradients since changing input x0 changes all the outputs,
            hence causing the cost to change. To make it more efficient, make sure to use softmax
            together with cross entropy to make the backpropogation algorithm faster.

            One intuition I gathered from doing this work about the jacobian and it's effect was thinking
            of another example where a vector in the input of a function (more than one variable). Ex:
            ShouldIGoOutToday(RainingPercentage, WindSpeed, Temperature). Now, the jacobian could tell us
            how much wind speed and temperature affect my decision to go out by making RainingPercentage static.
            Hence, ShouldIGoOutToday(0, WindSpeed, Temperature). The jacobian gives you all the gradients for
            the vectorized function (maybe need a better explanation, sry). The jacobian gives you answers
            on how some input changes the output directly or indirectly.

            Now for this function, I derived the jacobian by calculating the derivative of the softmax function
            in certain conditions. For the diagonal, it became (-1) * Si * Sj when i = j, where Si is the output node i which
            was computed by S(input I, rest Of Inputs) where input I was the target. The off-diagonal became Si * (1 - Sj) when i != j.
            Now having the jacobian, we want to understand how much input node I changes the cost. Since Input I changes all output nodes, it's
            indirect effect on the cost function has to be included, which means that for input i, we can't just look at the derivative dsi/dxi,
            we also need to look at dsj/dxi, dsy/dxi and so on. Hence we add up the total effect that the variable xi, or input i, has on the cost.
            So we are going to, for each input, multiply every gradient with the corresponding output derivative like this: dSi/dxj * dC/dSi, and then
            summing them all up.

            :param derivatives: The incoming derivatives from the layer in front.
            :type derivatives: numpyp.ndarray

            :return: Returns the propogation derivative for use in the layers in the back.
            :rtype: numpyp.ndarray
        """
        # Convert the outputs to a column vector.
        outputAsColumn: np.ndarray = self._outputs[:, None]

        # Faster way to calculate the jacobian.
        jacobian: np.ndarray = np.diagflat(self._outputs) - outputAsColumn @ np.transpose(outputAsColumn)

        toProp: np.ndarray = jacobian @ derivatives

        return toProp