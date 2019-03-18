import deep_nn.deep_nn_utils as nnu
import numpy as np
from unittest import TestCase
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestDeepNNUtils(TestCase):
    def test_softmax_one_dimention(self):
        assert_array_almost_equal(nnu.softmax([2, 3, 5, 6])[0], [0.01275478, 0.03467109, 0.25618664, 0.69638749])

    def test_softmax_two_dimentions(self):
        input = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array([[0.04742587, 0.04742587, 0.04742587], [0.95257413, 0.95257413, 0.95257413]])
        assert_array_almost_equal(nnu.softmax(input)[0], expected)
