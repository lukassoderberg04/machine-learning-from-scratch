from __future__ import annotations
from pathlib import Path
import pickle

from nn.network import Network

class Memory():
    """
        The Memory class takes care of saving and loading the models.
    """
    
    def __init__(self: "Memory") -> None:
        pass

    def SaveNetwork(self, network: Network, path: str) -> None:
        if not isinstance(network, Network):
            raise TypeError("SaveNetwork expects a Network instance!")

        file_path = Path(path).resolve()

        # Ensure directory exists!
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(network, f, protocol=pickle.HIGHEST_PROTOCOL)

    def LoadNetwork(self, path: str) -> Network:
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Network file does not exist: {file_path}!")

        with open(file_path, "rb") as f:
            network = pickle.load(f)

        if not isinstance(network, Network):
            raise TypeError("Loaded object is not a Network instance!")

        return network