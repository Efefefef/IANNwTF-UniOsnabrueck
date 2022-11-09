import numpy as np
from matplotlib import pyplot as plt

def create_dataset():
    x = np.random.rand(100)
    x = x.reshape(x.shape[0], 1)
    t = x**2
    # plt.scatter(x, t, s=5)
    # plt.show()
    return (x, t)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0) * 1

def mse(y, t):
    return (y - t)**2 / 2

def d_mse(y, t):
    return y - t

class Layer:
    def __init__(self, n_inputs, n_units, output_layer=False):
        self.biases = np.zeros((n_units, 1))
        self.weights = np.random.rand(n_inputs, n_units)
        self.n_units = n_units
        self.output_layer = output_layer
        self.input, self.preactivation, self.activation = None, None, None

    def forward_step(self, input):
        self.input = input
        self.input = self.input.reshape(self.input.shape[0], 1)
        self.preactivation = self.weights.T @ self.input + self.biases
        self.activation = relu(self.preactivation)
        # print('FORWARD STEP:')
        # print('Input: ', self.input)
        # print('Weights: ', self.weights)
        # print('Biases: ', self.biases)
        # print('Preactivation: ', self.preactivation)
        # print('Activation: ', self.activation)
        return self.activation
    
    def backward_step(self, output, learning_rate):
        gradient_loss_over_activation = d_mse(output, self.activation) if self.output_layer else output
        gradient_activation_over_preactivation = d_relu(self.preactivation)
        gradient_preactivation_over_weights = self.input
        gradient_preactivation_over_bias = np.ones((self.n_units, 1))
        gradient_preactivation_over_input = self.weights

        # print('\n Backward step:')
        gradient_loss_over_preactivation = gradient_activation_over_preactivation * gradient_loss_over_activation
        gradient_loss_over_weights = gradient_preactivation_over_weights @ gradient_loss_over_preactivation.T
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
    epochs = 1000
    losses = []
    for e in range(epochs):
        for (i, j) in zip(x, t):
            y = mlp.forward_step(i)
            loss = mse(y, j)
            losses.append(loss)
            mlp.backpropagation(loss)
        print(f"Epoch: {e + 1}, Loss: {losses[-1]}")

    predictions = []
    for a in x:
        y = mlp.forward_step(a)
        predictions.append(y)
    plt.scatter(x, predictions)
    plt.show()


if __name__ == "__main__":
    main()


