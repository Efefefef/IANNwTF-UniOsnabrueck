#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import copy 

#create data set
def create_dataset():
    x = np.random.rand(100)
    x = np.sort(x)
    t = x**3-x**2

    #plot to show how the function is supposed to look like
    #plt.plot(x,t)
    #plt.show()

    return (x,t)
    

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
        self.biases = np.zeros(n_units)

        self.layer_input = None
        self.preactivation = None
        self.layer_activation = None
        

    def forward_step(self, input):

        self.layer_input = input
        self.preactivation = np.matmul(self.layer_input, self.weight_matrix)  + self.biases
        self.layer_activation = relu(self.preactivation)

        return self.layer_activation
        
    #output layer doesnt have activation function
    def forward_step_output(self, input):

        self.layer_input = input
        self.preactivation = np.matmul(self.layer_input, self.weight_matrix) + self.biases
        self.layer_activation = copy.deepcopy(self.preactivation)

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

    #derivative of no activation function (identity function) is just 1
    def backward_step_output(self, gradient_activation):
        
        de_preact_times_grad_act = np.multiply(self.preactivation, gradient_activation)
        
        input_T = np.transpose(self.layer_input)


        gradient_weights = np.matmul(input_T, de_preact_times_grad_act)
        gradient_bias = de_preact_times_grad_act

        weight_T = np.transpose(self.weight_matrix)
        preact_times_activation = np.matmul(self.preactivation,np.asarray([gradient_activation]))
        gradient_input = np.matmul(preact_times_activation, weight_T)
        
        #update weights and biases
        self.weight_matrix = self.weight_matrix - self.learning_rate * gradient_weights
        self.biases = self.biases - self.learning_rate * gradient_bias

        return gradient_input

class MLP():
    def __init__(self) -> None:
        self.hidden_layer = Layer(0.001,10,1)
        self.output_layer = Layer(0.001,1,10)

    def forward_step(self, input):
        hidden_layer_output = self.hidden_layer.forward_step(input)
        output = self.output_layer.forward_step_output(hidden_layer_output)

        return output

    def backpropagation(self,gradient_activation):
        gradient_input = self.output_layer.backward_step_output(gradient_activation)
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
    epoch_size = 100
    
    x,t = create_dataset()
    mlp = MLP()

    
    mean_loss = []
    for e in range(epoch_size):

        losses = training(mlp,x,t)
        mean_loss.append(np.mean(losses))

    #plot the mean loss per epoch
    plt.plot(range(epoch_size),mean_loss)
    plt.show()



    #see if it works
    y = np.ndarray(shape = x.shape)
    for i in range(x.shape[0]):
        y[i] = mlp.forward_step(np.expand_dims(np.asarray([x[i]]), axis = 0))


    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    main()