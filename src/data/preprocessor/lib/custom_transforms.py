import random
import numpy as np
from PIL import Image

class RandomRotate90:

    def __init__(self, allow_clockwise=True, allow_counter_clockwise=True, *args, **kwargs):
        self.rotators = []
        self.rotators.append(RandomRotate90.rotate_clockwise_90) if allow_clockwise else None
        self.rotators.append(RandomRotate90.rotate_counter_clockwise_90) if allow_counter_clockwise else None

    def __call__(self, img):
        reconvert = False   
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            reconvert = True
        result = random.choice(seq=self.rotators)(img)
        if reconvert:
            result = Image.fromarray(result)
        return result

    @classmethod
    def rotate_clockwise_90(cls, np_img):
        """
        90-deg clockwise rotation to make the slide vertical
        Series of 2 operations - transpose row and col dimensions, invert vals in each row
        """
        np_result = np.transpose(np_img, [1, 0, 2])
        np_result = np_result[:,::-1,:]
        return np_result

    @classmethod
    def rotate_counter_clockwise_90(cls, np_img):
        """
        90-deg clockwise rotation to make the slide vertical
        Series of 2 operations - transpose row and col dimensions, invert vals in each column
        """
        np_result = np.transpose(np_img, [1, 0, 2])
        np_result = np_result[::-1,:,:]
        return np_result
