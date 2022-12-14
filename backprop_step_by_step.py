#backprop step by step

import numpy as np


# Relu activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of relu
def d_relu(x):
    return (x > 0) * 1

# Mean squared error
def mse(y, t):
    return (y - t)**2 / 2

# Derivative of the mean squared error
def d_mse(y, t):
    return y - t

class Layer:
    def __init__(self, n_inputs, n_units, biases, weights):
        self.biases = biases
        self.weights = weights
        self.n_units = n_units
        self.input, self.preactivation, self.activation = None, None, None

    def forward_step(self, input):
        print("Forward step:")
        print("self.weights: {}".format(self.weights))
        print("self.biases: {}".format(self.biases))
        self.input = input
        print("input: {}".format(self.input))

        self.preactivation = self.weights.T @ self.input + self.biases
        self.activation = relu(self.preactivation)

        print("preactivation: {}".format(self.preactivation))
        print("activation: {}".format(self.activation))

        return self.activation
    
    def backward_step(self, output, learning_rate):
        print("Backward step:")

        gradient_loss_over_activation = output
        print("gradient_loss_over_activation: {}".format(gradient_loss_over_activation))

        gradient_activation_over_preactivation = d_relu(self.preactivation)
        print("gradient_activation_over_preactivation/d_relu(preactiv): {}".format(gradient_activation_over_preactivation))

        gradient_preactivation_over_weights = self.input
        print("gradient_preactivation_over_weights/layer input: {}".format(gradient_preactivation_over_weights))

        gradient_preactivation_over_bias = np.ones((self.n_units, 1))
        print("gradient_preactivation_over_bias/np.ones((self.units, 1)): {}".format(gradient_preactivation_over_bias))
        
        gradient_preactivation_over_input = self.weights
        print("gradient_preactivation_over_input/self.weights: {}".format(gradient_preactivation_over_input))

        gradient_loss_over_preactivation = gradient_activation_over_preactivation * gradient_loss_over_activation
        gradient_loss_over_weights = gradient_preactivation_over_weights @ gradient_loss_over_preactivation.T
        gradient_loss_over_bias = gradient_preactivation_over_bias.T @ gradient_loss_over_preactivation
        gradient_loss_over_input = gradient_preactivation_over_input @ gradient_loss_over_preactivation

        print("gradient_loss_over_preactivation: {}".format(gradient_loss_over_preactivation))
        print("gradient_loss_over_weights: {}".format(gradient_loss_over_weights))
        print("gradient_loss_over_bias: {}".format(gradient_loss_over_bias))
        print("gradient_loss_over_input: {}. Will be given to next layer".format(gradient_loss_over_input))

        
        self.weights -= gradient_loss_over_weights * learning_rate
        print("new weights: {}".format(self.weights))

        self.biases -= gradient_loss_over_bias * learning_rate
        print("new biases: {}".format(self.biases))

        return gradient_loss_over_input

class MLP:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward_step(self, input):
        for layer in self.layers:
            input = layer.forward_step(input)
        return input

    def backpropagation(self, output):
        for layer in reversed(self.layers):
            output = layer.backward_step(output, self.learning_rate)

    def train(self, x, t):

        y = self.forward_step(x)
        loss = mse(y, t)
        print("prediction: {}".format(y))
        print("target: {}".format(t))
        print("loss: {}".format(loss))
        dmse_yt = d_mse(y, t)
        print("derivative of mean squared error of pred y and target t: {}".format(dmse_yt))
        self.backpropagation(dmse_yt)
        return loss


def main():
    learning_rate = 0.1

    input_size = 3
    output_size = 1 #without bias
    hidden_layer1_size = 2 #without bias

    weights_input = np.asarray([[1., -0.5],[0.5,-5.],[0.5,-5.]]) #layer size x next layer size 
    weights_hidden1 = np.asarray([[-0.5],[1]]) #layer size x next layer size 
    bias_input = np.asarray([[-2.], [6.]]) #next layer size x 1
    bias_hidden1_units = np.asarray([[2.]]) #next layer size x 1

    mlp = MLP([Layer(input_size, hidden_layer1_size, bias_input, weights_input), 
                Layer(hidden_layer1_size, output_size, bias_hidden1_units, weights_hidden1)], 
                learning_rate)
    
    input = [[4.],[5.],[-3.]]  #more values like [[1],[2],[3]]
    target = [[0.2]]  #more values like [[4],[5],[6]]

    if weights_input.shape != np.zeros((input_size,hidden_layer1_size)).shape:
        print("WRONG DIMENSIONS IN INPUT WEIGHTS")
    if weights_hidden1.shape != np.zeros((hidden_layer1_size,output_size)).shape:
        print("WRONG DIMENSIONS IN HIDDEN LAYER 1 WEIGHTS")
    if bias_input.shape != np.zeros((hidden_layer1_size,1)).shape:
        print("WRONG DIMENSIONS IN INPUT BIAS")
    if bias_hidden1_units.shape != np.zeros((output_size,1)).shape:
        print("WRONG DIMENSIONS IN HIDDEN LAYER 1 BIAS")


    loss = mlp.train(input, target)



if __name__ == "__main__":
    main()


