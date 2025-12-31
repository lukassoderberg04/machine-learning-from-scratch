from __future__ import annotations

from layer import Layer

class Network():
    """
        A base class for defining a neural network.
    """

    def __init__(self: "Network") -> None:
        self._layers: list[Layer] | None = None

    def CheckLayerConnection(self) -> None:
        """
            Checks that the layer is fully connected, else throws an error!
        """
        if self._layers is None:
            raise RuntimeError("Layers of a neural network has not been initialized yet!")
        
        if len(self._layers <= 0):
            raise RuntimeError("A network can't have 0 layers!")
        
        for (index, layer) in enumerate(self._layers[:-1]):
            layer_output_size: int = layer.GetSize()[1]

            next_layer: Layer = self._layers[index + 1]
            next_layer_input_size: int = next_layer.GetSize()[0]

            if layer_output_size != next_layer_input_size:
                raise RuntimeError(f"Layers don't fully connect! Between layer: {index} and {index + 1}!")