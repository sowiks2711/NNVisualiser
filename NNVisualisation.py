from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
import time
import seaborn
from matplotlib.animation import FuncAnimation
import sys
import json


class Neuron():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)
        return circle


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, weights):
        self.weights = weights
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer + 1
        self.previous_layer = self.__get_previous_layer(network)
        self.network = network
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons + 1)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (
                self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, thickness):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
                             linewidth=thickness * 2,
                             color=(0.5 - min(0, thickness) / 2, 0.5 + max(0, thickness) / 2, 0))
        pyplot.gca().add_line(line)
        return line

    def draw(self, layerType=0):
        artists = []
        i = 0
        j = 0
        if layerType != -1:
            artists.append(self.neurons[0].draw(self.neuron_radius))
        for neuron in self.neurons[1:]:
            #print(len(self.neurons[1:]))
            j = 0
            artists.append(neuron.draw(self.neuron_radius))
            if self.previous_layer:

                for previous_layer_neuron in self.previous_layer.neurons:
                    #print(len(self.previous_layer.neurons))

                    artists.append(self.__line_between_two_neurons(neuron, previous_layer_neuron,
                                                                   self.weights[i + j * len(self.neurons[1:])]))
                    j += 1
            i += 1
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer ' + str(layerType), fontsize=12)


class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer, biggest_weight):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.biggest_weight = biggest_weight
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, weights):
        norm_weigths = [w / self.biggest_weight for w in weights]
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, norm_weigths)
        self.layers.append(layer)

    def draw(self):
        artists = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            artists.append(layer.draw(i))
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Neural Network architecture', fontsize=15)
        return artists


class DrawNN():
    def __init__(self, input_size, weights):
        self.neural_network = [input_size]
        for weight in weights:
            self.neural_network.append(int(len(weight) / (self.neural_network[-1] + 1)))
        self.weights = weights
        self.weights.insert(0, [])

    def draw(self):
        widest_layer = max(self.neural_network)
        biggest_weight = max([max([abs(x) for x in layer]) for layer in self.weights if layer != []])
        network = NeuralNetwork(widest_layer, biggest_weight)
        for l, w in zip(self.neural_network, self.weights):
            network.add_layer(l, w)
        return network.draw()


class NNVisualisation():
    def __init__(self, input_size, weights):
        self.input_size = input_size
        self.old_weights = weights
        self.new_weights = weights
        self.list_weights = None
        self.fig = pyplot.figure()

    def __init__(self, input_size, weights, biases):
        self.input_size = input_size
        self.old_weights = None
        self.new_weights = None
        self.new_biases = None
        self.list_weights = weights
        self.list_biases = biases
        self.fig = pyplot.figure()

    def update(self, i):
        pyplot.gca().clear()
        old_weight_mod = 1.0 - i * 0.1
        new_weight_mod = i * 0.1

        modded_weights = []
        for old_layer, new_layer in zip(self.old_weights, self.new_weights):
            modded_layer = []
            for old_weight, new_weight in zip(old_layer, new_layer):
                modded_layer.append(old_weight * old_weight_mod + new_weight * new_weight_mod)
            modded_weights.append(modded_layer)

        network = DrawNN(self.input_size, modded_weights)
        return network.draw()

    def update_lists(self, i):
        pyplot.gca().clear()
        self.old_weights = self.list_weights[i]
        self.new_weights = self.list_weights[i]
        self.new_biases = self.list_biases[i]
        modded_weights = []
        for bias_layer, new_layer in zip(self.new_biases, self.new_weights):
            modded_layer = []
            for bias in bias_layer:
                modded_layer.append(bias)
            for new_weight in new_layer:
                modded_layer.append(new_weight)
            modded_weights.append(modded_layer)

        network = DrawNN(self.input_size, modded_weights)
        return network.draw()

    def draw(self, new_weights):
        self.new_weights = new_weights
        anim = FuncAnimation(self.fig, self.update, frames=np.arange(0, 10), interval=200)
        pyplot.show()
        self.old_weights = new_weights

    def draw(self):
        anim = FuncAnimation(self.fig, self.update_lists, frames=np.arange(0, len(self.list_weights)), interval=500)
        pyplot.show()


class NNVisualisationAdaptedFactory:
    def createVisualisator(self, input_size, L, parameters):
        return NNVisualisationAdapted(input_size, L, parameters)

    def createVisualisatorList(self, input_size, L, parameters_list):
        return NNVisualisationAdapted(input_size, L, parameters_list, None)


class NNVisualisationAdapted(NNVisualisation):

    def __init__(self, input_size, L, parameters):
        weights = self.extract_weights_to_array_form(parameters, L)
        NNVisualisation.__init__(self, input_size, weights)

    def __init__(self, input_size, L, parameters, pass_var):
        old_list_weights = []
        list_weights = []
        old_list_biases = []
        list_biases = []
        for param in parameters:
            old_list_weights.append(self.extract_weights_to_array_form_from_list(param, L))
            old_list_biases.append(self.extract_biases_to_array_form_from_list(param, L))

        for i in range(0, len(old_list_weights)):
            list_weights.append([])
            list_biases.append([])
            for j in range(0, len(old_list_weights[i])):
                list_weights[i].append([item for sublist in old_list_weights[i][j] for item in sublist])
            for j in range(0, len(old_list_biases[i])):
                list_biases[i].append([item for sublist in old_list_biases[i][j] for item in sublist])
        NNVisualisation.__init__(self, input_size, list_weights, list_biases)

    def draw(self, parameters, L):
        new_weights = self.extract_weights_to_array_form(parameters, L)
        NNVisualisation.draw(self, new_weights)

    def draw(self):
        NNVisualisation.draw(self)

    def extract_weights_to_array_form(self, parameters, L):
        weights = []
        for i in range(1, L):
            weights.append(parameters["W" + str(i)].reshape(-1).tolist())
        return weights

    def extract_weights_to_array_form_from_list(self, parameters, L):
        weights = []
        for i in range(1, L):
            weights.append(parameters["W" + str(i)])
        return weights

    def extract_biases_to_array_form_from_list(self, parameters, L):
        weights = []
        for i in range(1, L):
            weights.append(parameters["b" + str(i)])
        return weights


if __name__ == "__main__":
    json_filename = sys.argv[1]
    with open(json_filename) as f: json_parameters = json.load(f)
    # vis = NNVisualisation(2, [[1, 2, 3, 3, 2, 1], [3, 2, 1, 1, 2, 3, 3, 2, 1], [1, 2, 3]])
    vis = NNVisualisationAdaptedFactory().createVisualisatorList(json_parameters[0], json_parameters[1],
                                                                 json_parameters[2:])
    weights = json_filename
    # vis = NNVisualisation(2,json_parameters[2:],None)
    vis.draw()
    # vis.draw([[10, 2, 3, 3, 2, 10], [3, 2, 10, -10, 2, 3, 3, 2, -10], [-10, 2, 3]])
