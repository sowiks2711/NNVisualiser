import numpy as np
from IPython.core.debugger import set_trace

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    Z = np.clip(Z, -500, 500)

    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def linear(Z):
    """
    Implement the linear function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.array(Z, copy=True)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache

def tanh(Z):
    """
    Implement the tanh function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """

    A = np.tanh(Z)

    assert (A.shape == Z.shape)

    cache = Z
    return A, cache


def softmax(Z):
    """
    Implement the softmax function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    cache = Z
    shiftx = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(shiftx)
    A = exps / np.sum(exps, axis=0, keepdims=True)

    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s, _ = sigmoid(Z)
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def linear_func_backward(dA, cache):
    """
    Implement the backward propagation for a single linear unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

    assert (dZ.shape == Z.shape)

    return dZ


def tanh_backward(dA, cache):
    """
    Implement the backward propagation for a single tanh unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache

    s = np.tanh(Z)
    dZ = dA * (1 - s**2)

    assert (dZ.shape == Z.shape)

    return dZ


def softmax_backward(dA, cache):

    Z = cache
    s, _ = softmax(Z)

    dZ = dA * s * (1. - s)

    return dZ


def quadratic_cost(AL, Y, m):
    diff = AL - Y
    cost = (1. / m) * np.sum(np.multiply(diff, diff))
    return cost


def categorical_crossentropy_cost(AL, Y, m):
    """
     Implement the categorical_crossentropy cost function.

     Arguments:
     AL -- probability vector corresponding to your label predictions, shape (categories, number of examples)
     Y -- one hotencoded true "label" vector

     Returns:
     cost -- cross-entropy cost
    """
    L = -np.sum(Y * np.log(AL), axis=0)
    J = (1. / m) * np.sum(L)

    return J


def binary_crossentropy(AL, Y, m):
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    cost = 1. / m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost
