# HW-02
# Syifa Hasanah (shasanah)
# Assignment: Multi-Layer Perceptron

from pprint import pprint

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

## Task 1 - Building dataset
def generate_data ():
    x = np.random.rand(100)
    t = x**3 - x**2
    return x,y

#plt.scatter(x,t)
#plt.show()

## Task 2 - Perceptrons
# Instruction for 'Layer' class
# - consrtuctor :
#   parameter : n_units (int) = n-unit in the layer L
#               input_units (int) = n-unit in layer (L-1)

def relu (z):
    """
    function for ReLU
    """
    z[z<0]=0

class Layer(object):
    def __init__(self, n_units, input_units):
        self.n_units = n_units
        self.input_units = input_units
        self.weights = np.random.rand(input_units, n_units)
        self.biases = np.zeros(n_units)
        self.layer_inp = None
        self.layer_preact = None
        self.layer_act = None
        self.learn_rate = None

    def forward_step(self, ):
        """
        calculating output by using ReLu
        as the activation function
        towards 'inp' as the input
        return: output = relu(preactivation)
        """
        self.layer_preact = np.matmul(np.transpose(self.layer_inp), (self.weights)) + self.biases # z .
        #^-- this should be calculated from layer+1 (?)
        self.layer_act = relu(self.layer_preact) # a
        return self.layer_act

    def backward_step(self, error_next_l):
        """
        updating unit's parameter (i.e. weights and bias)
        return:
        """
        sigma_der_preact = self.layer_preact[self.layer_preact>0]=1
        error_l = np.multiply(sigma_der_preact, error_next_l)
        grd_biases = error_l
        grd_weights = np.matmul(np.transpose(self.layer_inp), error_l)
        grd_inp= np.matmul(error_l, np.transpose(self.weights))
        self.biases = self.biases - (self.learn_rate * grd_biases)
        self.weights = self.weights - (self.learn_rate * grd_weights)
        return grd_inp

layer1 = Layer(n_units=3, input_units=1)
print('\nCHECK LAYER 1:')
pprint(vars(layer1))

layer1.layer_inp = np.array([10])
print('\nLAYER 1, AFTER INPUT:')
pprint(vars(layer1))

layer1.forward_step()
print('\nLAYER 1, AFTER FORWARD STEP:')
pprint(vars(layer1))

#layer1.forward_step()


# # Task 03 - Multi-Layer Perceptron
# class MLP(object):
#     """
#     combining 'Layer' class into a MLP
#     """
#     def __init__(self, layer):
#         #
#     def forward_step(self):
#         """
#         passing input to entire network
#         """
#         return
#
#     def backpropagation(self):
#         """
#         updating all the weights and biases in network given a loss value
#         """
#         return

## Task 04 - Training
# MSE = to compute network's loss
# 1) Create MLS (1 hidden layer of 10 units, input layer of 1 unit, output layer of 1 unit)
# 2) Train MLP for 1000 epochs
#