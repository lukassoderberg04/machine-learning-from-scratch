from mnist.mnist_dataloader import MnistDataloader
from pathlib import Path
import unittest

class TestDataloaderInitialization(unittest.TestCase):
    def test_wrong_format(self) -> None:
        # Arrange:
        batchSizeNegative: int = -1
        wrongPath: str = "fjagagwejvdkv"

        currentFilePath: Path = Path(__file__).resolve()
        csvFilePath: Path = currentFilePath.parent.parent / "data" / "mnist_test.csv"

        # Assert:
        self.assertRaises(FileNotFoundError, MnistDataloader, wrongPath, 10)
        self.assertRaises(TypeError, MnistDataloader, csvFilePath, batchSizeNegative)

class TestDataloaderLoading(unittest.TestCase):
    currentFilePath: Path = Path(__file__).resolve()
    csvFilePath: Path = currentFilePath.parent.parent / "data" / "mnist_test.csv"

    def test_batch_loading(self) -> None:
        # Arrange:
        batchSize: int = 10

        # Act:
        loader: MnistDataloader = MnistDataloader(self.csvFilePath, batchSize)
        batch: list[MnistDataloader.DataPair] = loader.ReadOneBatch()
        del loader

        # Assert:
        self.assertTrue(len(batch) == batchSize, msg=f"The batch size was not as expected. Was instead: {len(batch)}")

    def test_loading_image_valid(self) -> None:
        # Act:
        loader: MnistDataloader = MnistDataloader(self.csvFilePath, 1)
        batch: list[MnistDataloader.DataPair] = loader.ReadOneBatch()
        
        # Assert:
        self.assertEqual(len(batch), 1)

if __name__ == "__main__":
    unittest.main()