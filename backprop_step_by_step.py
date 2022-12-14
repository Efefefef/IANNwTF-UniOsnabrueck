#backprop step by step

import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

# Relu activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of relu
def d_relu(x):
    return (x > 0) * 1

def linear(x):
    return x

def d_linear(x):
    print("d linear: {}".format(np.ones_like(x)))
    return np.ones_like(x)


# Mean squared error
def mse(y, t):
    sum = 0
    elements = 0
    y=y
    for i in range(y.shape[0]):
        print(i, y.shape)
        print(y[i][0])
        sum += (t[i][0] - y[i][0])**2
        elements +=1

    #doenst divide by n, returns one number
    #print("mse: {}".format((sum)))
    #return np.asarray([[sum]])

    #if 1/n(y-t)**2 is needed, returns one number
    #print("mse: {}".format((sum/elements)))
    #return np.asarray([[sum/elements]])

    #if we want loss for every output:
    return (y - t)**2 / 2

# Derivative of the mean squared error
def d_mse(y, t):

    #sum = 0
    #elements = 0
    #y=y.T
    #for i in range(y.shape[1]):
    #    sum += (y[0][i] - t[i][0])
    #    elements +=1

    #without dividing
    #print("d_mse: {}".format((2*sum)))
    #return np.asarray([[(2*sum)]])

    # if 2/n(y-t) is needed, returns one number 
    #print("d_mse: {}".format((2*sum/elements)))
    #return np.asarray([[(2*sum/elements)]])

    #if we want loss for every output:
    return np.asarray(y-t)


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
        self.activation = linear(self.preactivation)

        print("preactivation: {}".format(self.preactivation))
        print("activation: {}".format(self.activation))

        return self.activation
    
    def backward_step(self, output, learning_rate):
        print("Backward step:")

        gradient_loss_over_activation = output
        print("gradient_loss_over_activation: {}".format(gradient_loss_over_activation))

        gradient_activation_over_preactivation = d_linear(self.preactivation)
        print("gradient_activation_over_preactivation/d_linear(preactiv): {}".format(gradient_activation_over_preactivation))

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

    input_size = 2
    output_size = 2 #without bias
    hidden_layer1_size = 2 #without bias

    weights_input = np.asarray([[0.5, -0.5],[-1, 2.]]) #layer size x next layer size. in one bracket weights OF ONE neuron 
    weights_hidden1 = np.asarray([[-1.,-0.5], [2, 1.]]) #layer size x next layer size 
    bias_input = np.asarray([[0.5], [3.]]) #next layer size x 1
    bias_hidden1_units = np.asarray([[2.],[3]]) #next layer size x 1

    mlp = MLP([Layer(input_size, hidden_layer1_size, bias_input, weights_input), 
                Layer(hidden_layer1_size, output_size, bias_hidden1_units, weights_hidden1)], 
                learning_rate)
    
    input = [[3.],[-1.]]  #more values like [[1],[2],[3]]
    target = [[0.],[2.]]  #more values like [[4],[5],[6]]

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


