from unittest import TestCase

from numpy.testing import assert_array_almost_equal

from deep_nn.deep_nn_model import ForwardPropagator
from deep_nn_tests.test_cases import linear_activation_forward_test_case, L_model_forward_test_case


class TestForwardPropagator(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.A_prev, cls.W, cls.b = linear_activation_forward_test_case()
        cls.X, cls.parameters = L_model_forward_test_case()

    def test_linear_activation_forward_sigmoid(self):
        forward_propagator = ForwardPropagator(None)
        A, linear_activation_cache = forward_propagator.linear_activation_forward(self.A_prev, self.W,
                                                                                  self.b, activation="sigmoid")

        assert_array_almost_equal(A, [[0.96890023, 0.11013289]])

    def test_linear_activation_forward_relu(self):
        forward_propagator = ForwardPropagator(None)
        A, linear_activation_cache = forward_propagator.linear_activation_forward(self.A_prev, self.W,
                                                                                  self.b, activation="relu")

        assert_array_almost_equal(A, [[3.43896131, 0.]])

    def test_L_model_forward(self):
        forward_propagator = ForwardPropagator(["relu", "relu", "sigmoid"])
        AL, caches = forward_propagator.L_model_forward(self.X, self.parameters)

        assert_array_almost_equal(AL, [[0.17007265, 0.2524272]])
