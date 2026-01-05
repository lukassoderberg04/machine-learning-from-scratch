from __future__ import annotations

from nn.network import Network
import tkinter as tk
import numpy as np
from mnist.mnist_dataloader import MnistDataloader

class MnistGui():
    """
        The GUI for checking how the model performs on the test data.
    """

    CELL_SIZE: int = 8  # Size in pixels of the cells.
    GRID_SIZE: int = 28 # Size of the grid (x * x).

    def __init__(self: "MnistGui", network: Network, dataloader: MnistDataloader) -> None:
        """
            :param network: The network of which to validate!
            :type network: nn.Network

            :param dataloader: The responsible object for loading the images.
            :type dataloader: mnist.MnistDataloader
        """
        self._network: Network            = network
        self._dataloader: MnistDataloader = dataloader

    def Run(self) -> None:
        """
            Fires up the GUI.
        """
        self._root = tk.Tk()
        self._root.title("Mnist model predictions")

        # Canvas to show the image.
        self._canvas = tk.Canvas(self._root, width=28*10, height=28*10, bg="white")
        self._canvas.grid(row=0, column=0, padx=10, pady=10)

        # Labels for true label and prediction.
        self._label_true_var = tk.StringVar(value="True Label: ?")
        self._label_true = tk.Label(self._root, textvariable=self._label_true_var, font=("Arial", 14))
        self._label_true.grid(row=1, column=0, pady=5)

        self._label_pred_var = tk.StringVar(value="Prediction: ?")
        self._label_pred = tk.Label(self._root, textvariable=self._label_pred_var, font=("Arial", 14))
        self._label_pred.grid(row=2, column=0, pady=5)

        # Button to load next image.
        self._btn_next = tk.Button(self._root, text="Next Image", command=self._showNextImage)
        self._btn_next.grid(row=3, column=0, pady=10)

        self._root.mainloop()

    def _showNextImage(self) -> None:
        """
            Loads the next image from the dataloader, displays it, and updates labels.
        """
        batch = self._dataloader.ReadOneBatch()
        if not batch:
            print("No more images to show.")
            return

        label, image = batch[0]  # Grab the first image in batch.

        img = np.array(image.GetPixels(), dtype=np.uint8).reshape(28, 28)

        # Scale up for display.
        from PIL import Image, ImageTk
        pilImg = Image.fromarray(img)
        pilImg = pilImg.resize((28*10, 28*10), Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(pilImg)

        # Update canvas.
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        input = image.GetNormalizedPixels()  # flat np.ndarray for network.
        output = self._network.Compute(input)
        predicted = int(np.argmax(output))

        # Update labels.
        self._label_true_var.set(f"True Label: {label}")
        self._label_pred_var.set(f"Prediction: {predicted}")