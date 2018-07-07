from PIL import Image
import numpy as np


class ProcessImage(object):

    def __init__(self, path, height, width, gray_scale=False):

        if gray_scale:

            self.image = np.array(Image.open(path).resize((height, width)).convert('LA').convert('RGB')).astype("float32")

        else:

            self.image = np.array(Image.open(path).resize((height, width))).astype("float32")

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

    @staticmethod
    def unprocess_image(image):

        image[:, :, 2] += 123.68

        image[:, :, 1] += 116.779

        image[:, :, 0] += 103.939

        image = image[:, :, ::-1]

        image = np.clip(image, 0, 255).astype("uint8")

        img = Image.fromarray(image, 'RGB')

        return img

    @staticmethod
    def histogram_match(image_1, image_2):

        height, width, channels = image_1.shape

        mu1 = np.mean(image_1, axis=(0, 1))

        reshaped1 = image_1.reshape((height * width, channels)).T

        vCov1 = np.cov(reshaped1, rowvar=True)

        chol1 = np.linalg.cholesky(vCov1)

        mu2 = np.mean(image_2, axis=(0, 1))

        reshaped2 = image_2.reshape((height * width, channels)).T

        vCov2 = np.cov(reshaped2, rowvar=True)

        chol2 = np.linalg.cholesky(vCov2)

        A = chol1.dot(np.linalg.inv(chol2))

        mu1 = mu1.reshape(3, 1)

        mu2 = mu2.reshape(3, 1)

        b = mu1 - A.dot(mu2)

        image2_prime = A.dot(reshaped2) + b

        image2_prime = image2_prime.T

        image2_prime = image2_prime.reshape(height, width, channels).astype("float32")

        return image2_prime


if __name__ == "__main__":

    test = ProcessImage("/Users/liam/Desktop/Projects/picsnet/photos/scream.jpg", 200, 200)

    test.process_image()



