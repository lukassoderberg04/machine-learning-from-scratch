from __future__ import annotations

from nn.network import Network
import tkinter as tk
import numpy as np

class App():
    """
        The GUI application that is used to draw digits yourself
        and see what the specified model (with the specified weights)
        predicts.
    """

    CELL_SIZE: int = 8  # Size in pixels of the cells.
    GRID_SIZE: int = 28 # Size of the grid (x * x).

    def __init__(self: "App", network: Network) -> None:
        """
            :param network: The network of which to validate!
            :type network: nn.Network
        """
        self._network: Network = network

        self._root: tk.Tk | None                 = None
        self._canvas: tk.Canvas | None           = None
        self._predictionVar: tk.StringVar | None = None

        # Grid state:
        self._grid: list[list[int]] = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

    def _drawCell(self, event: tk.Event) -> None:
        """
            Draws a cell under the mouse.

            :param event: The mouse event.
            :type event: tk.Event
        """
        assert self._canvas is not None

        x: int = event.x // self.CELL_SIZE
        y: int = event.y // self.CELL_SIZE

        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            if self._grid[y][x] == 0:
                self._grid[y][x] = 1
                self._canvas.create_rectangle(
                    x * self.CELL_SIZE,
                    y * self.CELL_SIZE,
                    (x + 1) * self.CELL_SIZE,
                    (y + 1) * self.CELL_SIZE,
                    fill="black",
                    outline=""
                )
                self._predict()

    def _clearGrid(self, event: tk.Event) -> None:
        """
            Clears the grid.
        
            :param event: The mouse event.
            :type event: tk.Event
        """
        assert self._canvas is not None
        assert self._predictionVar is not None

        self._canvas.delete("all")

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                self._grid[y][x] = 0

        self._predictionVar.set("Prediction: ?")
        self._drawGridLines()

    def _drawGridLines(self) -> None:
        """
            Draw the grid lines.
        """
        assert self._canvas is not None

        for i in range(self.GRID_SIZE + 1):
            self._canvas.create_line(
                0, i * self.CELL_SIZE,
                self.GRID_SIZE * self.CELL_SIZE, i * self.CELL_SIZE,
                fill="lightgray"
            )
            self._canvas.create_line(
                i * self.CELL_SIZE, 0,
                i * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE,
                fill="lightgray"
            )

    def _predict(self) -> None:
        """
            Predicts the number using the model.
        """
        assert self._predictionVar is not None

        # Convert grid to a numpy array.
        inputArray: np.ndarray = np.array(
            [
                float(self._grid[y][x])  # invert: black=0.0, white=1.0!
                for y in range(self.GRID_SIZE)
                for x in range(self.GRID_SIZE)
            ],
            dtype=np.float64
        )

        try:
            output: np.ndarray = self._network.Compute(inputArray)

            # If Compute returns class probabilities
            if hasattr(output, "__len__"):
                prediction = int(np.argmax(output))
            else:
                prediction = int(output)

            self._predictionVar.set(f"Prediction: {prediction}")
        except Exception:
            self._predictionVar.set("Prediction: error")

    def Run(self) -> None:
        """
            Fires up the GUI.
        """
        self._root = tk.Tk()
        self._root.title("Machine Learning Digits")

        # Create the canvas.
        self._canvas = tk.Canvas(
            self._root,
            width=self.CELL_SIZE * self.GRID_SIZE,
            height=self.CELL_SIZE * self.GRID_SIZE,
            bg="white"
        )
        self._canvas.grid(row=0, column=0, padx=10, pady=10)

        # Prediction label.
        self._predictionVar = tk.StringVar(value="Prediction: ?")
        predictionLabel: tk.Label = tk.Label(
            self._root,
            textvariable=self._predictionVar,
            font=("Arial", 16)
        )
        predictionLabel.grid(row=1, column=0, pady=10)

        # Draw grid lines once.
        self._drawGridLines()

        # Mouse bindings.
        self._canvas.bind("<Button-1>", self._drawCell)
        self._canvas.bind("<B1-Motion>", self._drawCell)
        self._canvas.bind("<Button-3>", self._clearGrid)

        self._root.mainloop()