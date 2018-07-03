from PIL import Image
import numpy as np


class ProcessImage(object):

    def __init__(self, path, height, width, gray_scale=False):

        if gray_scale:

            self.image = np.array(Image.open(path).resize((height, width))).astype("float32")

        else:

            self.image = np.array(Image.open(path).resize((height, width)).convert('LA').convert('RGB')).astype("float32")

        self.means = [123.68, 116.779, 103.939]

        self.processed_image = None

    def __subtract_means__(self):

        self.processed_image = self.image

        for i, mean in enumerate(self.means):

            self.processed_image[:, :, i] -= mean

    def __expand_dims__(self):

        self.processed_image = np.expand_dims(self.processed_image, 0)

    def __flip_channels__(self):

        self.processed_image = self.processed_image[:, :, :, ::-1]

    def process_image(self):

        self.__subtract_means__()

        self.__expand_dims__()

        self.__flip_channels__()


if __name__ == "__main__":

    test = ProcessImage("/Users/liam/Desktop/Projects/picsnet/photos/scream.jpg", 200, 200)

    test.process_image()



