import numpy as np
from matplotlib import pyplot as plt

def create_dataset():
    # Generate random numbers between 0 and 1
    x = np.random.rand(100)

    # Calculating targets
    t = x**3 - x**2

    # Plotting the data points
    # plt.scatter(x, t, s=5)
    # plt.show()
    return (x, t)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0) * 1

def loss(y, t):
    return (y - t)**2 / 2

def d_loss(y, t):
    return y - t

class Layer:
    def __init__(self, n_inputs, n_units, output_layer=False):
        self.biases = np.zeros(n_units)
        self.weights = np.random.rand(n_inputs, n_units) * 2 - 1
        self.output_layer = output_layer
        self.input, self.preactivation, self.activation = None, None, None

    def forward_step(self, input):
        self.input = input
        self.preactivation = self.input @ self.weights + self.biases
        self.activation = relu(self.preactivation)
        return self.activation
    
    def backward_step(self, output, learning_rate):
        print('layer')
        gradient_loss_over_activation = d_loss(output, self.activation) if self.output_layer else output
        gradient_activation_over_preactivation = d_relu(self.preactivation)
        gradient_preactivation_over_weights = self.input
        gradient_preactivation_over_bias = np.ones(self.preactivation.shape)
        gradient_preactivation_over_input = self.weights
        
        gradient_loss_over_preactivation = gradient_loss_over_activation * gradient_activation_over_preactivation
        print(gradient_loss_over_preactivation.shape, gradient_preactivation_over_weights.shape, gradient_preactivation_over_input.shape, gradient_preactivation_over_bias.shape)
        gradient_loss_over_weights = gradient_preactivation_over_weights.T @ gradient_loss_over_preactivation
        gradient_loss_over_bias = gradient_preactivation_over_bias.T @ gradient_loss_over_preactivation
        gradient_loss_over_input = gradient_preactivation_over_input @ gradient_loss_over_preactivation

        self.weights -= gradient_loss_over_weights * learning_rate
        self.biases -= gradient_loss_over_bias * learning_rate
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


def main():
    x, t = create_dataset()
    mlp = MLP([Layer(1, 10), Layer(10, 10), Layer(10, 1, output_layer=True)], learning_rate=0.01)
    epochs = 100
    for i in range(epochs):
        y = mlp.forward_step(np.array([x]).T)
        mlp.backpropagation(y - t)
        print(f"Epoch: {i}, Loss: {np.mean(loss(y, t))}")


if __name__ == "__main__":
    main()


