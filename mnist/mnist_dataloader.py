from __future__ import annotations

from io import TextIOWrapper
from pathlib import Path
from mnist.mnist_image import MnistImage
import csv

class MnistDataloader:
    """
        Loads data from the mnist dataset.
    """

    # Class types.
    Label    = int
    DataPair = tuple["MnistDataloader.Label", MnistImage]

    def __init__(self, pathToDataset: str, batchSize: int = 10) -> None:
        """
            :param pathToDataset: Description
            :type pathToDataset: str

            :param batchSize: Description
            :type batchSize: int

            :raises TypeError: If the batchSize is negative or zero.
            :raises FileNotFoundError: If the pathToDataset does not exist.
        """

        absPath = Path(pathToDataset).resolve()

        if (not absPath.exists()):
            raise FileNotFoundError(f"The dataset file does not exist: {absPath}")
        
        self._path: Path = absPath

        self._file: TextIOWrapper = open(self._path, "r")

        self._csvFile = csv.reader(self._file)

        if (batchSize > 1):
            self._batchSize: int = batchSize
        else:
            raise TypeError("Batch size can't be lower than 1!")

        return
    
    def ReadOneBatch(self) -> list["MnistDataloader.DataPair"]:
        """
            Reads one batch of data pairs from the dataset.
        
            :return: List of successfully read data pairs.
            :rtype: list[MnistDataloader.DataPair]
        """

        pairs: list[MnistDataloader.DataPair] = []
        readCount: int = 0 # Successfully count of read pairs from the csv file.

        for _ in range(self._batchSize):
            pair: MnistDataloader.DataPair | None = self._readOneDataPair()

            if (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], MnistDataloader.Label) and isinstance(pair[1], MnistImage)):
                readCount += 1 # Increment counter.

                pairs.append(pair)

        return pairs
    
    def _readOneDataPair(self) -> "MnistDataloader.DataPair" | None:
        """
            Reads one data pair from the mnist csv file.
            
            :return: A data pair.
            :rtype: MnistDataloader.DataPair | None
        """

        nextLine: list[str] | None = self._readNextLine()

        if (nextLine is None): 
            return None

        convertedLine: list[int] = [int(val) for val in nextLine]

        # 1 is for the label. 28 * 28 is for the image size.
        if (len(convertedLine) != 1 + (28 * 28)):
            # Format doesn't match expected, return None!
            return None
        
        label: MnistDataloader.Label = convertedLine[0] # The first item is the label.

        imagePixels: list[int] = convertedLine[1:] # Take everything except for the first element.
        
        return (label, MnistImage(imagePixels))
    
    def _readNextLine(self) -> list[str] | None:
        """
            Tries to read the next line in the csv file.
        
            :return: The list of values as strings or None if there was no line to read.
            :rtype: list[str] | None
        """

        try:
            return next(self._csvFile)
        except StopIteration:
            return None
        
    def __del__(self):
        self._file.close()