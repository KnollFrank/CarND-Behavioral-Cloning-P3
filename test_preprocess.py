from unittest import TestCase

import numpy as np

from preprocess import flip_image, flip_measurement, flip_images, flip_measurements, \
    get_augmented_images_and_measurements, get_steering_left, get_steering_right


class FlipTestCase(TestCase):

    def test_flip_image(self):
        # GIVEN

        # WHEN
        flipped_image = flip_image(np.array([[1., 0., 0.],
                                             [0., 2., 0.],
                                             [0., 0., 3.]]))

        # THEN
        np.testing.assert_array_equal(flipped_image,
                                      np.array([[0., 0., 1.],
                                                [0., 2., 0.],
                                                [3., 0., 0.]]))

    def test_flip_measurement(self):
        # GIVEN

        # WHEN
        flipped_measurement = flip_measurement(40)

        # THEN
        self.assertEqual(flipped_measurement, -40)

    def test_steering_left(self):
        # GIVEN
        steering_center = -30

        # WHEN
        steering_left = get_steering_left(steering_center)

        # THEN
        self.assertEqual(steering_left, steering_center + 0.2)

    def test_steering_right(self):
        # GIVEN
        steering_center = -20

        # WHEN
        steering_right = get_steering_right(steering_center)

        # THEN
        self.assertEqual(steering_right, steering_center - 0.2)

    def test_flip_images(self):
        # GIVEN

        # WHEN
        flipped_images = list(flip_images([np.array([[1., 0., 0.],
                                                     [0., 2., 0.],
                                                     [0., 0., 3.]]),
                                           np.array([[10., 0., 0.],
                                                     [0., 20., 0.],
                                                     [0., 0., 30.]])]))

        # THEN
        np.testing.assert_array_equal(flipped_images[0],
                                      np.array([[0., 0., 1.],
                                                [0., 2., 0.],
                                                [3., 0., 0.]]))
        np.testing.assert_array_equal(flipped_images[1],
                                      np.array([[0., 0., 10.],
                                                [0., 20., 0.],
                                                [30., 0., 0.]]))

    def test_flip_measurements(self):
        # GIVEN

        # WHEN
        flipped_measurements = list(flip_measurements([40, 25.3]))

        # THEN
        self.assertEqual(flipped_measurements, [-40, -25.3])

    def test_get_augmented_images_and_measurements(self):
        # GIVEN
        images = [np.array([[1., 0., 0.],
                            [0., 2., 0.],
                            [0., 0., 3.]])]

        measurements = [40]

        # WHEN
        augmented_images, augmented_measurements = get_augmented_images_and_measurements(images, measurements)

        # THEN
        np.testing.assert_array_equal(augmented_images[0], images[0])
        np.testing.assert_array_equal(augmented_images[1], flip_image(images[0]))
        self.assertEqual(augmented_measurements, [40, -40])
