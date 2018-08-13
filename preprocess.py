import cv2
from keras.layers import Cropping2D

factor = 1.0 / 4.0


def resize(image):
    return cv2.resize(image, dsize=None, fx=factor, fy=factor)


def preprocess(image):
    return resize(image)


def create_Cropping2D():
    return Cropping2D(cropping=((int(70 * factor), int(25 * factor)), (0, 0)))


def get_input_shape():
    return (int(160 * factor), int(320 * factor), 3)
