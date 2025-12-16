from mnist.mnist_image import MnistImage
import unittest
import random

class TestImageInitialization(unittest.TestCase):
    def test_wrong_format(self) -> None:
        # Arrange:
        shortImage: list[int] = [x for x in range(100)]
        longImage: list[int]  = [x for x in range((28 * 28) + 1)]

        # Assert:
        self.assertRaises(TypeError, MnistImage, shortImage, msg="Didn't throw error when initializing with a smaller image size.")
        self.assertRaises(TypeError, MnistImage, longImage, msg="Didn't throw error when initializing with a bigger image size.")

class TestImageFunctionality(unittest.TestCase):
    def test_normalized_pixel_data(self) -> None:
        # Arrange:
        imageData: list[int] = [random.randint(0, 255) for x in range(28 * 28)]

        # Act:
        image: MnistImage = MnistImage(imageData)

        # Assert:
        self.assertTrue(all(0 <= x <= 1 for x in image.GetNormalizedPixels()), msg="Not all values were normalized!")

if __name__ == "__main__":
    unittest.main()