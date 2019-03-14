from unittest import TestCase

from numpy.testing import assert_array_almost_equal

from deep_nn.deep_nn_model import BackwardPropagator
from deep_nn_tests.test_cases import linear_backward_test_case, linear_activation_backward_test_case, \
    L_model_backward_test_case


class TestBackwardPropagator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dZ, cls.linear_cache = linear_backward_test_case()
        cls.AL_for_linear_backpropagation, cls.linear_activation_cache = linear_activation_backward_test_case()
        cls.AL_for_L_layer_backpropagation, cls.Y_assess, cls.caches = L_model_backward_test_case()

    def test_linear_backward(self):
        backward_propagator = BackwardPropagator(["relu", "relu", "sigmoid"], "binary_crossentropy")
        dA_prev, dW, db = backward_propagator.linear_backward(self.dZ, self.linear_cache)

        assert_array_almost_equal(dA_prev,
                                  [[0.51822968, -0.19517421],
                                   [-0.40506361, 0.15255393],
                                   [2.37496825, -0.89445391]])
        assert_array_almost_equal(dW, [[-0.10076895, 1.40685096, 1.64992505]])
        assert_array_almost_equal(db, [[0.50629448]])

    def test_linear_activation_backward_sigmpoid(self):
        backward_propagator = BackwardPropagator(["relu", "relu", "sigmoid"], "binary_crossentropy")
        dA_prev, dW, db = backward_propagator.linear_activation_backward(self.AL_for_linear_backpropagation,
                                                                         self.linear_activation_cache,
                                                                         activation="sigmoid")

        assert_array_almost_equal(dA_prev,
                                  [[0.11017994, 0.01105339],
                                   [0.09466817, 0.00949723],
                                   [-0.05743092, -0.00576154]])

        assert_array_almost_equal(dW,
                                  [[0.10266786, 0.09778551, -0.01968084]])
        assert_array_almost_equal(db, [[-0.05729622]])

    def test_linear_activation_backward_relu(self):
        backward_propagator = BackwardPropagator(["relu", "relu", "sigmoid"], "binary_crossentropy")
        dA_prev, dW, db = backward_propagator.linear_activation_backward(self.AL_for_linear_backpropagation,
                                                                         self.linear_activation_cache,
                                                                         activation="relu")

        assert_array_almost_equal(dA_prev,
                                  [[0.44090989, 0.],
                                   [0.37883606, 0.],
                                   [-0.2298228, 0.]])
        assert_array_almost_equal(dW,
                                  [[0.44513824, 0.37371418, -0.10478989]])
        assert_array_almost_equal(db, [[-0.20837892]])

    def test_L_model_backwards(self):
        backward_propagator = BackwardPropagator(["relu", "relu", "sigmoid"], "binary_crossentropy")
        grads = backward_propagator.L_model_backward(self.AL_for_L_layer_backpropagation, self.Y_assess, self.caches)

        assert_array_almost_equal(grads['dW1'], [[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                                                 [0., 0., 0., 0.],
                                                 [0.05283652, 0.01005865, 0.01777766, 0.0135308]])
        assert_array_almost_equal(grads['db1'],
                                  [[-0.22007063],
                                   [0.],
                                   [-0.02835349]])
        assert_array_almost_equal(grads['dA1'], [[0., 0.52257901],
                                                 [0., -0.3269206],
                                                 [0., -0.32070404],
                                                 [0., -0.74079187]])

