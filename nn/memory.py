from __future__ import annotations

from nn.network import Network

class Memory():
    """
        The Memory class takes care of saving and loading the models.
    """
    
    def __init__(self: "Memory") -> None:
        pass

    def SaveNetwork(self, network: Network, path: str) -> None:
        pass

    def LoadNetwork(self, path: str) -> Network:
        pass

""" SAVING:
    import pickle

    # Suppose `network` is your Network instance
    network = Network()
    network._learningRate = 0.01
    # ... setup layers, cost, etc.

    # Save to a file
    with open("network.pkl", "wb") as f:
        pickle.dump(network, f)
"""

""" LOADING:
    import pickle

    with open("network.pkl", "rb") as f:
        loaded_network = pickle.load(f)
"""