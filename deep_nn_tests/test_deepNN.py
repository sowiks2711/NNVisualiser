from unittest import TestCase
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
from deep_nn.deep_nn import initialize_parameters, random_mini_batches, update_parameters_with_momentum, \
    linear_forward, linear_activation_forward, L_model_forward, compute_cost, linear_backward, \
    linear_activation_backward, L_model_backward
from deep_nn_tests.test_cases import random_mini_batches_test_case, update_parameters_with_momentum_test_case, \
    linear_forward_test_case, linear_activation_forward_test_case, L_model_forward_test_case, compute_cost_test_case, \
    linear_backward_test_case, linear_activation_backward_test_case, L_model_backward_test_case


# noinspection PyPep8Naming
class TestDeepNN(TestCase):
    def test_initialize_parameters_deep(self):
        parameters = initialize_parameters([5, 4, 3])
        expected_W1 = np.array(
            [[0.799899, 0.195213, 0.043155, -0.833379, -0.124052],
             [-0.158653, -0.037003, -0.280403, -0.019596, -0.213418],
             [-0.587578, 0.395615, 0.394137, 0.764544, 0.022376],
             [-0.180977, -0.243892, -0.691606, 0.439328, -0.492412]])
        expected_b1 = [[0.], [0.], [0.], [0.]]
        expected_W2 = [[-0.592523, -0.102825,  0.743074,  0.118358],
                       [-0.511893, -0.356497,  0.312622, -0.080257],
                       [-0.384418, -0.115015,  0.372528,  0.988055]]
        expected_b2 = [[0.], [0.], [0.]]
        assert_array_almost_equal(parameters["W1"], expected_W1)
        assert_array_almost_equal(parameters["b1"], expected_b1)
        assert_array_almost_equal(parameters["W2"], expected_W2)
        assert_array_almost_equal(parameters["b2"], expected_b2)

    def test_random_mini_batches(self):
        X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
        mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

        self.assertEqual(mini_batches[0][0].shape, (12288, 64))
        self.assertEqual(mini_batches[1][0].shape, (12288, 64))
        self.assertEqual(mini_batches[2][0].shape, (12288, 20))
        self.assertEqual(mini_batches[0][1].shape, (1, 64))
        self.assertEqual(mini_batches[1][1].shape, (1, 64))
        self.assertEqual(mini_batches[2][1].shape, (1, 20))
        assert_array_almost_equal(
           mini_batches[0][0][0][0:3],
           [0.90085595, -0.7612069, 0.2344157]
        )

    def test_update_parameters_with_momentum(self):
        parameters, grads, v = update_parameters_with_momentum_test_case()
        parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)

        assert_array_almost_equal(parameters['W1'],
                                  [[1.62544598, -0.61290114, -0.52907334],
                                  [-1.07347112,  0.86450677, -2.30085497]])
        assert_array_almost_equal(parameters['b1'], [[1.74493465], [-0.76027113]])
        assert_array_almost_equal(parameters['W2'],
                                  [[0.31930698, -0.24990073, 1.4627996],
                                   [-2.05974396, -0.32173003, -0.38320915],
                                   [1.13444069, -1.0998786, -0.1713109]])
        assert_array_almost_equal(parameters['b2'], [[-0.87809283], [0.04055394], [0.58207317]])
        assert_array_almost_equal(v["dW1"],
                                  [[-0.11006192, 0.11447237, 0.09015907],
                                   [0.05024943, 0.09008559, -0.06837279]])
        assert_array_almost_equal(v["db1"], [[-0.01228902], [-0.09357694]])
        assert_array_almost_equal(v["dW2"],
                                  [[-0.02678881, 0.05303555, -0.06916608],
                                   [-0.03967535, -0.06871727, -0.08452056],
                                   [-0.06712461, -0.00126646, -0.11173103]])
        assert_array_almost_equal(v["db2"], [[0.02344157], [0.16598022], [0.07420442]])

    def test_linear_forward(self):
        A, W, b = linear_forward_test_case()

        Z, linear_cache = linear_forward(A, W, b)

        assert_array_almost_equal(Z, [[ 3.26295337, -1.23429987]])

    def test_linear_activation_forward(self):
        A_prev, W, b = linear_activation_forward_test_case()
        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
        assert_array_almost_equal(A, [[0.96890023, 0.11013289]])

        A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
        assert_array_almost_equal(A, [[3.43896131, 0.]])

    def test_L_model_forward(self):
        X, parameters = L_model_forward_test_case()
        AL, caches = L_model_forward(X, parameters)
        assert_array_almost_equal(AL, [[0.17007265, 0.2524272]])

    def test_compute_cost(self):
        Y, AL = compute_cost_test_case()
        actual_cost = compute_cost(AL, Y)
        self.assertEqual(actual_cost, 0.41493159961539694)

    def test_linear_backward(self):
        dZ, linear_cache = linear_backward_test_case()
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

        assert_array_almost_equal(dA_prev,
                                  [[0.51822968, -0.19517421],
                                   [-0.40506361,  0.15255393],
                                   [2.37496825, -0.89445391]])
        assert_array_almost_equal(dW, [[-0.10076895, 1.40685096, 1.64992505]])
        assert_array_almost_equal(db, [[0.50629448]])

    def test_linear_activation_backward(self):
        AL, linear_activation_cache = linear_activation_backward_test_case()

        dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="sigmoid")

        assert_array_almost_equal(dA_prev,
                                  [[0.11017994, 0.01105339],
                                   [0.09466817, 0.00949723],
                                   [-0.05743092, -0.00576154]])

        assert_array_almost_equal(dW,
                                  [[0.10266786, 0.09778551, -0.01968084]])
        assert_array_almost_equal(db, [[-0.05729622]])

        dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation="relu")

        assert_array_almost_equal(dA_prev,
                                  [[0.44090989, 0.],
                                   [0.37883606, 0.],
                                   [-0.2298228, 0.]])
        assert_array_almost_equal(dW,
                                  [[0.44513824, 0.37371418, -0.10478989]])
        assert_array_almost_equal(db, [[-0.20837892]])

    def test_L_model_backwards(self):
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = L_model_backward(AL, Y_assess, caches)

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

    def test_L_model_backwards_explicit_activations(self):
        AL, Y_assess, caches = L_model_backward_test_case()
        grads = L_model_backward(AL, Y_assess, caches, ["relu", "sigmoid"])

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
