import unittest

from nn.layers.relu import Relu
from nn.layer import Layer
import numpy as np

class TestReluComputation(unittest.TestCase):
    def test_computes_correct_forward(self) -> None:
        # Arrange:
        relu: Layer            = Relu(3)
        inputs1: np.array      = np.array([4.0, 3.0, -1.0])
        inputs2: np.array      = np.array([1.0, -10.0, -15.0])

        # Act:
        outputs1: np.array = relu.Forward(inputs1)
        outputs2: np.array = relu.Forward(inputs2)

        # Assert:
        np.testing.assert_array_equal(outputs1, np.array([4.0, 3.0, 0.0]))
        np.testing.assert_array_equal(outputs2, np.array([1.0, 0.0, 0.0]))

    def test_computes_correct_backward(self) -> None:
        # Arrange:
        relu: Layer            = Relu(3)
        inputs1: np.array      = np.array([4.0, 3.0, -1.0])
        inputs2: np.array      = np.array([1.0, -10.0, -15.0])
        derivatives1: np.array = np.array([1.0, -1.0, -2.0])
        derivatives2: np.array = np.array([-1.0, 1.0, 1.0])

        # Act:
        relu.Forward(inputs1)
        toProp1: np.array = relu.Backward(derivatives1)
        relu.Forward(inputs2)
        toProp2: np.array = relu.Backward(derivatives2)

        # Assert:
        np.testing.assert_array_equal(toProp1, np.array([1.0, -1.0, 0.0]))
        np.testing.assert_array_equal(toProp2, np.array([-1.0, 0.0, 0.0]))

if __name__ == "__main__":
    unittest.main()