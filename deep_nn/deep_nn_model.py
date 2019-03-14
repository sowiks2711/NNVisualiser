import math

from NNVisualisation import VisualisatorFactory
from deep_nn.deep_nn import initialize_parameters
import numpy as np

from deep_nn.deep_nn_utils import softmax, tanh, linear, relu, sigmoid


class SequentialBuilder:
    def __init__(self):
        self.layers_dims = []
        self.layers_activations = []

    def addDense(self, nr_of_neurons, activation='relu'):
        self.layers_dims.append(nr_of_neurons)
        self.layers_activations.append(activation)

    def compile(self, loss, visualisation = False):
        visualisator_factory = None
        if visualisation:
             visualisator_factory  = VisualisatorFactory()
        return DeepNN(self.layers_dims, self.layers_activations, loss, visualisator_factory=visualisator_factory)


class DeepNN:
    def __init__(self, forward_propagator_factory, backward_propagator_factory, layers_dims, layers_activations, loss, visualisator_factory=None):
        self.layers_dims = layers_dims
        self.layers_activations = layers_activations
        self.loss = loss
        self.parameters = initialize_parameters(self.layers_dims)
        self.v = self.initialize_velocity(self.parameters)
        self.forward_propagator = forward_propagator_factory.create(layers_activations)
        self.backward_propagator = backward_propagator_factory.create(layers_activations)
        self.visualisator = None
        if visualisator_factory is not None:
            self.initialize_visualisator(layers_dims, visualisator_factory)

    def fit(self, X, Y, learning_rate=0.01, momentum=0.9, num_epochs=10000, mini_batch_size=64, verbose=True):
        L = len(self.layers_dims)  # number of layers in the neural networks
        costs = []  # to keep track of the cost
        seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
        num_outputs = Y.shape[0]

        # Optimization loop
        for i in range(num_epochs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                AL, caches = L_model_forward(minibatch_X, parameters, self.layers_activations)

                # Compute cost
                cost = compute_cost(AL, minibatch_Y, self.loss)

                # Backward propagation
                grads = L_model_backward(AL, minibatch_Y, caches, self.layers_activations, self.loss)

                # Update parameters
                parameters, v = update_parameters_with_momentum(parameters, grads, v, momentum, learning_rate)

            # Print the cost every 1000 epoch
            if verbose and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost))
            if i % 100 == 0:
                costs.append(cost)

        return parameters, costs

    def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
        """

        np.random.seed(seed)
        m = X.shape[1]
        num_outputs = Y.shape[0]
        mini_batches = []

        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((num_outputs, m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def initialize_velocity(parameters):
        """
        Initializes the velocity as a python dictionary with:
                    - keys: "dW1", "db1", ..., "dWL", "dbL"
                    - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
        Arguments:
        parameters -- python dictionary containing your parameters.
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl

        Returns:
        v -- python dictionary containing the current velocity.
                        v['dW' + str(l)] = velocity of dWl
                        v['db' + str(l)] = velocity of dbl
        """

        L = len(parameters) // 2  # number of layers in the neural networks
        v = {}

        for l in range(L):
            v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
            v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

        return v

    def initialize_visualisator(self, layers_dims, visualisator_factory):
        if visualisator_factory is not None:
            weights = self.extract_weights_to_array_form(self.parameters, len(layers_dims))
            self.visualisator = visualisator_factory.createVisualisator(layers_dims[0], self.parameters)

    def extract_weights_to_array_form(self, parameters, L):
        weights = []
        for i in range(1, L):
            weights.append(parameters["W" + str(i + 1)].reshape(-1).tolist())
        return weights


class ForwardPropagator:
    def __init__(self, layers_activations):
        self.layers_activations = layers_activations

    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            A, activation_cache = relu(Z)

        elif activation == "linear":
            A, activation_cache = linear(Z)

        elif activation == "tanh":
            A, activation_cache = tanh(Z)

        elif activation == "softmax":
            A, activation_cache = softmax(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters, layers_activations=None):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        num_outputs = parameters["W" + str(L)].shape[0]

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            current_activation_func = "relu"
            if layers_activations is not None:
                current_activation_func = layers_activations[l]
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                                 current_activation_func)
            caches.append(cache)

        last_activation_func = "sigmoid"
        if layers_activations is not None:
            last_activation_func = layers_activations[L]
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                              last_activation_func)
        caches.append(cache)

        assert (AL.shape == (num_outputs, X.shape[1]))

        return AL, caches

class ForwardPropagator:
    def __init__(self, layers_activations, parameters):
        self.paramaeters = parameters
        self.layers_activations = layers_activations

    def linear_forward(A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            A, activation_cache = relu(Z)

        elif activation == "linear":
            A, activation_cache = linear(Z)

        elif activation == "tanh":
            A, activation_cache = tanh(Z)

        elif activation == "softmax":
            A, activation_cache = softmax(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters, layers_activations=None):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        num_outputs = parameters["W" + str(L)].shape[0]

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            current_activation_func = "relu"
            if layers_activations is not None:
                current_activation_func = layers_activations[l]
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)],
                                                 parameters["b" + str(l)], current_activation_func)
            caches.append(cache)

        last_activation_func = "sigmoid"
        if layers_activations is not None:
            last_activation_func = layers_activations[L]
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                              last_activation_func)
        caches.append(cache)

        assert (AL.shape == (num_outputs, X.shape[1]))

        return AL, caches
