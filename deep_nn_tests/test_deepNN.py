from unittest import TestCase
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
from deep_nn.deep_nn import initialize_parameters_deep, random_mini_batches, update_parameters_with_momentum
from deep_nn_tests.test_cases import random_mini_batches_test_case, update_parameters_with_momentum_test_case


# noinspection PyPep8Naming
class TestDeepNN(TestCase):
    def test_initialize_parameters_deep(self):
        parameters = initialize_parameters_deep([5, 4, 3])
        expected_W1 = np.array(
            [[0.01788628, 0.0043651, 0.00096497, -0.01863493, -0.00277388],
             [-0.00354759, -0.00082741, -0.00627001, -0.00043818, -0.00477218],
             [-0.01313865, 0.00884622, 0.00881318,  0.01709573, 0.00050034],
             [-0.00404677, -0.0054536, -0.01546477, 0.00982367, -0.01101068]])
        expected_b1 = [[0.], [0.], [0.], [0.]]
        expected_W2 = [[-0.01185047, -0.0020565,  0.01486148,  0.00236716],
                       [-0.01023785, -0.00712993,  0.00625245, -0.00160513],
                       [-0.00768836, -0.00230031,  0.00745056,  0.01976111]]
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



