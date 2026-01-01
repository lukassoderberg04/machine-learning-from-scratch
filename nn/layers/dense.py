from __future__ import annotations

from nn.layer import Layer
import numpy as np

class Dense(Layer):
    """
        A dense layer is a layer where all input neruons are
        connected to all output neurons.
    """

    def __init__(
            self: "Dense", 
            input: int, 
            output: int, 
            useBias: bool = True
        ):
        """
            :param input: The amount of inputs.
            :type input: int

            :param output: The amount of outputs.
            :type output: int

            :param useBias: Tells if you want to use bias when calculating the output.
            :type useBias: bool
        """

        self._size: tuple[int, int] = (input, output)

        # Initialized to None since no forward pass has happened.
        self._inputs: np.array | None  = None
        self._outputs: np.array | None = None

        # Initialize the bias, if bias are to be used!
        self._usesBias: bool        = useBias
        self._bias: np.array | None = None
        if useBias:
            self._bias = np.zeros(shape=output)

        # Initialize the weights close to 0!
        self._weights: np.ndarray = np.random.normal(loc=0, scale=0.01, size=(output, input))

    def Forward(self, inputs: np.array) -> np.array:
        """
            For each output, calculate the inputs scaled by the weight summed up together,
            hence this formula: listOfInputs[] * listOfWeightsIntoOutput[] + bias.

            :param inputs: The incoming inputs. Has to be the correct size as defined in the constructor.
            :type inputs: numpy.array

            :return: The output, calculated by the formula explained above.
            :rtype: numpy.array
        """
        if len(inputs) != self.GetInputSize():
            raise RuntimeError("The input size was not as defined by the layer when forwarding!")
        
        self._inputs = inputs # Store the input as history.

        # Matrix multiplication. Will, for each output, take the dot product between each weight and input.
        # The dot products between a list of weights for a certain output and the input can look like this:
        # input: [a0, a1], weights: [w0, w1], dot product: a0 * w0 + a1 * w1.
        self._outputs = self._weights @ inputs

        if self._usesBias:
            self._outputs += self._bias # Will element wise add the bias if it was enabled.
        
        return self._outputs
    
    def Backward(self, derivatives: np.array) -> np.array:
        """
            Recieves derivatives from the front to help in computing the local
            derivative of this layer. This layer is going to calculate the derivatives
            in terms of the local weights and biases and change them accordingly.

            :param derivatives: The derivatives from the front layers, has to match the output size.
            :type derivatives: numpy.array

            :return: Returns the computed local derivative sum for each input node for use by the layers in the back.
            :rtype: numpy.array
        """
        if self._outputs is None:
            raise RuntimeError("There hasn't been a forward pass for this dense layer!")
        
        if len(derivatives) != self.GetOutputSize():
            raise RuntimeError("The derivatives doesn't match the output size when running backpropagation!")
        
        # Since the output function of this layer is y = input * weight + bias, the local derivative of
        # dy / dweight => input. And since the chain rule is present we'll multiply the forward derivative
        # with dy / dweight to get the local gradient for each weight. This is used to move the weights in a certain direction.
        weightGradients: np.ndarray = np.outer(derivatives, self._inputs)

        # Since dy / dbias => 1, it's just going to be derivatives since multiplying with 1 does NOTHING!
        # This is used to move the bias in a certain direction.
        biasGradients: np.array = derivatives

        # The propagation derivative is the sum of the weight derivative for each input.
        # The transpose function makes sure that each row corresponds to the weights going out from the corresponding input.
        # Then the dot product just sums up all the derivatives going out from the input. This is the derivatives that are
        # propagated backwards. Why it very much looks like the weightGradient, we are here instead calculating
        # dy / dinput, which is why we multiply by the weights instead of the input this time. This is because the input
        # to this system depends on variables calculated in the layers before this, hence why dy / dinput is calculated here.
        propagationDerivatives: np.array = np.transpose(self._weights) @ derivatives

        # Change the weights. Since each row fo the weightGradients contain the gradients for the weights arriving at the output
        # we can just elementwise take a step in the opposite direction. Imagine if the gradient for a weight is positive, that means
        # that the cost function is increasing if we continue in that step, hence we want to go the other way around (positive gradient * negative
        # becomes negative, hence moving "backwards"). If the gradient is instead negative, that means that we should take a step forward since 
        # it's going down, hence why we multiply by -1 since -gradient * -1 becomes positive, meaning we continue forward in the step of descent.
        self._weights -= weightGradients

        # Same for bias as well, but only if we have activated it.
        if (self._usesBias):
            self._bias -= biasGradients

        return propagationDerivatives