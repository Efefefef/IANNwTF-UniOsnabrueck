#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

#create data set
def create_dataset():
    x = np.random.rand(100)
    t = x**3-x**2 + 1
    #plot to show how the function is supposed to look like
    plt.scatter(x,t)
    plt.ylabel("target")
    plt.title("The function to be approximated")
    plt.show()

    return x,t
    
#shuffles dataset
def shuffle(x,t):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    x = x[indices]
    t = t[indices]

    return x,t

def relu(x):
    return np.maximum(0,x)

#derivative of relu
def de_relu(x):
    return (x > 0) * 1

class Layer():
    def __init__(self, learning_rate, n_units, n_input) -> None:
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.n_input = n_input
        self.weight_matrix = np.random.rand(self.n_input, self.n_units) * 2 - 1
        #self.biases = np.zeros(n_units)
        self.biases = np.random.rand(self.n_units) * 0.2 - 0.1 #worked better than with only zeros
        self.layer_input = None
        self.preactivation = None
        self.layer_activation = None
        

    def forward_step(self, input):

        self.layer_input = input
        self.preactivation = np.matmul(self.layer_input, self.weight_matrix)  + self.biases
        self.layer_activation = relu(self.preactivation)

        return self.layer_activation
        

    def backward_step(self, gradient_activation):
        
        de_relu_preact = np.asarray(de_relu(self.preactivation))
        de_preact_times_grad_act = np.multiply(de_relu_preact, gradient_activation)
        
        input_T = np.transpose(self.layer_input)

        gradient_weights = np.matmul(input_T, de_preact_times_grad_act)
        gradient_bias = de_preact_times_grad_act

        weight_T = np.transpose(self.weight_matrix)
        gradient_input = np.matmul(de_preact_times_grad_act, weight_T)

        #update weights and biases
        self.weight_matrix = self.weight_matrix - self.learning_rate * gradient_weights
        self.biases = self.biases - self.learning_rate * gradient_bias

        return gradient_input



class MLP():
    def __init__(self, learning_rate=0.01) -> None:
        self.hidden_layer = Layer(learning_rate,10,1)
        self.output_layer = Layer(learning_rate,1,10)

    def forward_step(self, input):
        hidden_layer_output = self.hidden_layer.forward_step(input)
        output = self.output_layer.forward_step(hidden_layer_output)

        return output

    def backpropagation(self,gradient_activation):
        gradient_input = self.output_layer.backward_step(gradient_activation)
        gradient_input = self.hidden_layer.backward_step(gradient_input)
        

def training(mlp,x,t):
    loss = np.zeros(shape = x.shape)
    for i in range(x.shape[0]):
        y = mlp.forward_step(np.expand_dims(np.asarray([x[i]]), axis = 0))
        loss[i] = 0.5 * ((y[0][0] - t[i])**2)
        activation_gradient = y[0][0] - t[i]
        mlp.backpropagation(activation_gradient)

    return loss



def main():
    epoch_size = 1000
    learning_rate = 0.01
    x,t = create_dataset()
    mlp = MLP(learning_rate)


    mean_loss = []
    for e in range(epoch_size):
        x,t = shuffle(x,t)
        losses = training(mlp,x,t)
        mean_loss.append(np.mean(losses))

    #plot the mean loss per epoch
    plt.plot(range(epoch_size),mean_loss)
    plt.ylabel("mean loss")
    plt.title("Mean loss per Epoch")

    plt.show()



    #see if it works, plots the predicted y-values
    y = np.ndarray(shape = x.shape)
    for i in range(x.shape[0]):
        y[i] = mlp.forward_step(np.expand_dims(np.asarray([x[i]]), axis = 0))


    plt.scatter(x,y)
    plt.ylabel("predicted y-value")
    plt.title("The learned function")

    plt.show()

if __name__ == "__main__":
    main()