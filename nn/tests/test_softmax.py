import unittest

from nn.layers.softmax import Softmax
from nn.layer import Layer
import numpy as np

class TestReluComputation(unittest.TestCase):
    def test_computes_correct_forward(self) -> None:
        # Arrange:
        softmax: Layer   = Softmax(3) # 3 inputs.
        inputs: np.array = np.array([2.0, 1.0, 0.1])

        # Act:
        outputs: np.array = softmax.Forward(inputs)

        # Assert:
        self.assertAlmostEqual(np.sum(outputs), 1.0, places=6) # All probabilities should sum up to 1.

    def test_computes_correct_backward(self) -> None:
        # Arrange:
        softmax: Layer                = Softmax(3)
        inputs: np.array              = np.array([1.0, 2.0, 3.0])
        incomingDerivatives: np.array = np.array([0.1, -0.2, 0.3])
        outputs = softmax.Forward(inputs)

        # Act:
        s: np.array          = outputs
        sCol: np.ndarray     = s[:, None] # Flip row array to column array.
        jacobian: np.ndarray = np.diagflat(s) - sCol @ np.transpose(sCol)

        expectedToProp: np.array = jacobian @ incomingDerivatives

        toProp: np.array = softmax.Backward(incomingDerivatives)

        # Assert:
        np.testing.assert_allclose(toProp, expectedToProp, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()