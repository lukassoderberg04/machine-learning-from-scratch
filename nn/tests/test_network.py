import unittest

from nn.layer import Layer
from nn.network import Network

class TestNetworkConnection(unittest.TestCase):
    def test_fully_connecting_layers(self) -> None:
        # Arrange:
        layer1: Layer = Layer()
        layer2: Layer = Layer()

        layer1._size = (3, 5) # 3 input, 5 output.
        layer2._size = (5, 1) # 5 input, 1 output.

        network: Network = Network()
        network._layers = [layer1, layer2]

        # Assert:
        try:
            network.CheckLayerConnection() # The test fails if an exception is thrown.
        except:
            self.fail("Layers should be connected but still fails the check!")

    def test_not_fully_connecting_layers(self) -> None:
        # Arrange:
        layer1: Layer = Layer()
        layer2: Layer = Layer()

        layer1._size = (3, 4) # 3 input, 5 output.
        layer2._size = (5, 1) # 5 input, 1 output.

        network: Network = Network()
        network._layers = [layer1, layer2]

        # Assert:
        try:
            network.CheckLayerConnection() # The test fails if an exception is thrown.
        except:
            return
        
        self.fail("Layers wasn't connected and still passed!")

if __name__ == "__main__":
    unittest.main()