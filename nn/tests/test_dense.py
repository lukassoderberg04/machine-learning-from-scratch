import unittest

from nn.layers.dense import Dense
from nn.layer import Layer
import numpy as np

class TestDenseComputation(unittest.TestCase):
    def test_computes_correct_forward(self) -> None:
        # Arrange:
        layer: Layer     = Dense(2, 3)
        layer._weights   = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        inputs: np.array = np.array([4.0, 3.0])

        # Act:
        outputs: np.array = layer.Forward(inputs)

        # Assert:
        np.testing.assert_array_equal(outputs, np.array([7.0, 7.0, 7.0]))

    def test_computes_correct_backward(self) -> None:
        # Arrange:
        layer: Layer          = Dense(2, 3)
        layer._weights        = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        inputs: np.array      = np.array([4.0, 3.0])
        derivatives: np.array = np.array([1.0, 1.0, 1.0])

        # Act:
        outputs: np.array     = layer.Forward(inputs)
        toPropagate: np.array = layer.Backward(derivatives)

        # Assert:
        self.assertEqual(len(toPropagate), 2) # Since the layer's input count is 2.
        np.testing.assert_array_equal(toPropagate, np.array([3.0, 3.0]))
        np.testing.assert_array_equal(layer._weights, np.array([[-3.0, -2.0], [-3.0, -2.0], [-3.0, -2.0]]))

if __name__ == "__main__":
    unittest.main()