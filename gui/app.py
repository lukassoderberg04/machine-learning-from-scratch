from __future__ import annotations

from nn.network import Network
import tkinter as tk

class App():
    """
        The GUI application that is used to draw digits yourself
        and see what the specified model (with the specified weights)
        predicts.
    """

    def __init__(self: "App", network: Network) -> None:
        """
            :param network: The network of which to validate!
            :type network: nn.Network
        """
        self._network: Network = network

    def Run(self) -> None:
        """
            Fires up the GUI.
        """

        CELL_SIZE: int = 40 # Size in pixels of the cells.
        GRID_SIZE: int = 28 # Size of the grid (x * x).

        root: tk.Tk = tk.Tk() # Creates a widget.
        root.title("Machine Learning Digits")

        # Create the canvas for the specified widget.
        canvas: tk.Canvas = tk.Canvas(root, width=CELL_SIZE * GRID_SIZE, height=CELL_SIZE * GRID_SIZE)
        canvas.grid(row=0, column=0, padx=10, pady=10)