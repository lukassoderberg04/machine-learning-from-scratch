import numpy

class MnistImage():
    """
        An image that reflects the data in the mnist dataset.
    """
    
    def __init__(self: "MnistImage", pixels: list[int]) -> None:
        """
            :param pixels: The values for each pixel. Length should be 28*28!
            :type pixels: list[int]
        """

        if (len(pixels) != 28 * 28):
            raise TypeError("Pixels was not of length 28*28!")
        
        # Clip the pixels so that each pixel value is between 0 and 255.
        self._pixels: list[int] = numpy.clip(pixels, 0, 255)

        # Normalize the pixel values to be between 0 and 1.
        self._normalizedPixels: list[float] = [float(val) / 255.0 for val in self._pixels]

        return
    
    def GetPixels(self) -> list[int]:
        """
            Get pixels of image.
            
            :return: A list of pixels with their value from 0 to 255.
            :rtype: list[int]
        """
        
        return self._pixels
    
    def GetNormalizedPixels(self) -> list[float]:
        """
            Get normalized pixels of image.
            
            :return: A list of pixels with their value from 0 to 1.
            :rtype: list[int]
        """

        return self._normalizedPixels