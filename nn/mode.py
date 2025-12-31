from enum import Enum

class Mode(Enum):
    """
        The current mode of a layer or neural network.
    """

    TRAINING   = 0
    PERFORMING = 1