import cv2
import numpy as np
import pandas as pd
from keras.layers import Cropping2D
from scipy import ndimage

factor = 1.0 / 4.0


def resize(image):
    return cv2.resize(image, dsize=None, fx=factor, fy=factor)


def preprocess(image):
    return resize(image)


def create_Cropping2D():
    return Cropping2D(cropping=((int(70 * factor), int(25 * factor)), (0, 0)))


def get_input_shape():
    return (int(160 * factor), int(320 * factor), 3)


def get_driving_log():
    base_path = '../CarND-Behavioral-Cloning-P3-data_from_udacity/data/'

    def correct_path(path):
        filename = path.split('/')[-1]
        return base_path + 'IMG/' + filename

    def correct_path_in_column(df, column):
        df[column] = df[column].map(correct_path)

    df = pd.read_csv(base_path + 'driving_log.csv')
    correct_path_in_column(df, 'center')
    correct_path_in_column(df, 'left')
    correct_path_in_column(df, 'right')
    return df


def get_images_and_measurements():
    def read_and_preprocess(image_file):
        return preprocess(ndimage.imread(image_file))

    def get_images_and_steerings(df, column, adapt_steering):
        images = df[column].map(read_and_preprocess).values.tolist()
        steerings = df['steering'].map(adapt_steering).values.tolist()
        return images, steerings

    df = get_driving_log()
    images_center, steerings_center = get_images_and_steerings(df, 'center', lambda x: x)
    images_left, steerings_left = get_images_and_steerings(df, 'left', get_steering_left)
    images_right, steerings_right = get_images_and_steerings(df, 'right', get_steering_right)
    return [*images_center, *images_left, *images_right], [*steerings_center, *steerings_left, *steerings_right]


def flip_image(image):
    return np.fliplr(image)


def flip_measurement(measurement):
    return -measurement


def flip_images(images):
    return map(flip_image, images)


def flip_measurements(measurements):
    return map(flip_measurement, measurements)


def get_steering_left(steering_center):
    return steering_center + 0.2


def get_steering_right(steering_center):
    return steering_center - 0.2

def get_augmented_images_and_measurements(images, measurements):
    return [*images, *flip_images(images)], [*measurements, *flip_measurements(measurements)]
