from unittest import TestCase

import numpy as np

from preprocess import flip


class FlipTestCase(TestCase):

    def test_flip(self):
        # GIVEN
        image = np.array([[1., 0., 0.],
                          [0., 2., 0.],
                          [0., 0., 3.]])

        measurement = 40

        # WHEN
        flipped_image, flipped_measurement = flip(image, measurement)

        # THEN
        np.testing.assert_array_equal(flipped_image,
                                      np.array([[0., 0., 1.],
                                                [0., 2., 0.],
                                                [3., 0., 0.]]))
        self.assertEqual(flipped_measurement, -measurement)
