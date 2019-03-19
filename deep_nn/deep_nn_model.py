import math

from NNVisualisation import NNVisualisationAdaptedFactory
from deep_nn.deep_nn_utils import binary_crossentropy, quadratic_cost, \
    categorical_crossentropy_cost
import numpy as np

from deep_nn.deep_nn_utils import softmax, tanh, linear, relu, sigmoid, relu_backward, sigmoid_backward, \
    linear_func_backward, tanh_backward, softmax_backward


class ForwardPropagatorFactory:
    def create(self, layers_activations):
        return ForwardPropagator(layers_activations)


class BackwardPropagatorFactory:
    def create(self, layers_activations, loss):
        return BackwardPropagator(layers_activations, loss)


class SequentialBuilder:
    def __init__(self):
        self.layers_dims = []
        self.layers_activations = []

    def add_dense(self, nr_of_neurons, activation='relu'):
        '''
        Adds neuron layer to model
        :param nr_of_neurons: integer number of neurons in layer
        :param activation: activation function used in neurons of this layer sigmoid | relu | tanh | softmax | linear
        '''
        self.layers_dims.append(nr_of_neurons)
        self.layers_activations.append(activation)

    def compile(self, loss, visualisation=False, disable_bias=False):
        '''
        Compiles nn model with layers added with add_dense method

        :param loss: function used to measure error MSE | binary_crossentropy | categorical_crossentropy
        :param visualisation: enables visualisation True | False
        :param disable_bias: disables bias neurons True | False
        :return: DeepNN object
        '''
        visualisator_factory = None
        if visualisation:
            visualisator_factory = NNVisualisationAdaptedFactory()
        return DeepNN(ForwardPropagatorFactory(),
                      BackwardPropagatorFactory(),
                      self.layers_dims,
                      self.layers_activations,
                      loss,
                      visualisator_factory,
                      disable_bias)


class DeepNN:
    def __init__(self, forward_propagator_factory, backward_propagator_factory,
                 layers_dims, layers_activations, loss, visualisator_factory=None,
                 disable_bias=False):
        self.layers_dims = layers_dims
        self.layers_activations = layers_activations
        self.loss = loss
        self.parameters = self.initialize_parameters(self.layers_dims)
        self.v = self.initialize_velocity(self.parameters)
        self.forward_propagator = forward_propagator_factory.create(layers_activations)
        self.backward_propagator = backward_propagator_factory.create(layers_activations, loss)
        self.visualisator_factory = visualisator_factory
        self.bias_factor = not disable_bias

    def fit(self, X, Y, learning_rate=0.01, momentum=0.9, num_epochs=10000, mini_batch_size=64, verbose=True):
        L = len(self.layers_dims)  # number of layers in the neural networks
        costs = []  # to keep track of the cost
        seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
        num_outputs = Y.shape[0]

        visualisator = None
        if self.visualisator_factory is not None:
            visualisator = self.visualisator_factory.createVisualisator(self.layers_dims[0], L, self.parameters)

        # Optimization loop
        for i in range(num_epochs):

            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = self.random_mini_batches(X, Y, mini_batch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                AL, caches = self.forward_propagator.L_model_forward(minibatch_X, self.parameters)

                # Compute cost
                cost = self.compute_cost(AL, minibatch_Y)

                # Backward propagation
                grads = self.backward_propagator.L_model_backward(AL, minibatch_Y, caches)

                # Update parameters
                self.update_parameters_with_momentum(grads, momentum, learning_rate)

            # Print the cost every 1000 epoch
            if verbose and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost))
            if i % 100 == 0:
                costs.append(cost)

            if visualisator is not None:
                visualisator.draw(self.parameters, L)

        return self.parameters, costs

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
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


    def initialize_parameters(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        parameters = {}
        L = len(layer_dims)  # number of layers in the network

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def initialize_velocity(self, parameters):
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

    def update_parameters_with_momentum(self, grads, beta, learning_rate):
        """
        Update parameters using Momentum

        Arguments:
        parameters -- python dictionary containing parameters:
                        parameters['W' + str(l)] = Wl
                        parameters['b' + str(l)] = bl
        grads -- python dictionary containing gradients for each parameters:
                        grads['dW' + str(l)] = dWl
                        grads['db' + str(l)] = dbl
        beta -- the momentum hyperparameter, scalar
        learning_rate -- the learning rate, scalar

        Returns:
        parameters -- python dictionary containing updated parameters
        v -- python dictionary containing updated velocities
        """

        L = len(self.parameters) // 2  # number of layers in the neural networks

        # Momentum update for each parameter
        for l in range(L):
            # compute velocities
            self.v["dW" + str(l + 1)] = beta * self.v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
            self.v["db" + str(l + 1)] = beta * self.v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]
            # update parameters
            self.parameters["W" + str(l + 1)] = self.parameters["W" + str(l + 1)] - learning_rate * self.v["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] = self.parameters["b" + str(l + 1)] - learning_rate * self.v["db" + str(l + 1)] * self.bias_factor

    def predict(self, X):
        """
        This function is used to predict the results of a  L-layer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        # Forward propagation
        probas, caches = self.forward_propagator.L_model_forward(X, self.parameters)

        return probas

    def predict_classes(self, X):
        m = X.shape[1]
        p = np.zeros((1, m))
        output_neurons = self.layers_dims[-1]
        probas = self.predict(X)

        if output_neurons == 1:
            p = (probas > 0.5)
        else:
            p = np.argmax(probas, axis=0)

        return p

    def compute_cost(self, AL, Y):
        """
        Implement the cost function.

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        epsilon = np.finfo(np.float32).eps
        AL[AL == 0] = epsilon
        AL[AL == 1] = 1 - epsilon

        if self.loss == "binary_crossentropy":
            cost = binary_crossentropy(AL, Y, m)
        elif self.loss == "MSE":
            cost = quadratic_cost(AL, Y, m)
        elif self.loss == "categorical_crossentropy":
            cost = categorical_crossentropy_cost(AL, Y, m)
        assert (cost.shape == ())

        return cost


class ForwardPropagator:
    def __init__(self, layers_activations):
        self.layers_activations = layers_activations
        self.activation_functions = {
            "sigmoid": sigmoid,
            "relu": relu,
            "linear": linear,
            "tanh": tanh,
            "softmax": softmax
        }

    def linear_forward(self, A, W, b):
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

        A, activation_cache = self.activation_functions[activation](Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, X, parameters):
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
            current_activation_func = self.layers_activations[l]
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                                                 current_activation_func)
            caches.append(cache)

        last_activation_func = self.layers_activations[L]
        AL, cache = self.linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)],
                                              last_activation_func)
        caches.append(cache)

        assert (AL.shape == (num_outputs, X.shape[1]))

        return AL, caches


class BackwardPropagator:
    def __init__(self, layers_activations, loss):
        self.layers_activations = layers_activations
        self.loss = loss
        self.activation_backward_functions = {
            "relu": relu_backward,
            "sigmoid": sigmoid_backward,
            "linear": linear_func_backward,
            "tanh": tanh_backward,
            "softmax": softmax_backward
        }

    def linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        dZ = self.activation_backward_functions[activation](dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches, cost_func="binary_crossentropy"):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid. lenear or softmax" (it's caches[L-1])

        Returns:
        grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        epsilon = np.finfo(np.float32).eps
        AL[AL == 0.] = epsilon
        AL[AL == 1.] = 1 - epsilon
        if self.loss == "binary_crossentropy":
            dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        elif self.loss == "MSE":
            dAL = AL - Y
        elif self.loss == 'categorical_crossentropy':
            dAL = AL - Y

        last_activation_func = self.layers_activations[L]

        # Lth layer (ACTIVATION_FUNC -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        computed_grads = self.linear_activation_backward(dAL, caches[L - 1], last_activation_func)
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = computed_grads

        for l in reversed(range(L - 1)):
            # lth layer: (ACTIVATION_FUNC -> LINEAR) gradients.
            # Inputs: "grads["dA" + str(l + 2)], caches, layers_activation[l]". Outputs: "grads["dA" + str(l + 1)],
            #                                                       grads["dW" + str(l + 1)],
            #                                                       grads["db" + str(l + 1)]
            current_activation_func = self.layers_activations[l]
            current_layer_grads = self.linear_activation_backward(grads["dA" + str(l + 2)], caches[l],
                                                                  current_activation_func)
            grads["dA" + str(l + 1)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = current_layer_grads

        return grads

