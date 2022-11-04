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

def mse(y, t):
    return (y - t)**2 / 2

def d_mse(y, t):
    return y - t

def add_dimension_if_one(x):
    if len(x.shape) == 1:
        x = np.array([x]).T
    return x

class Layer:
    def __init__(self, n_inputs, n_units, output_layer=False):
        self.biases = np.zeros(n_units)
        self.weights = np.random.rand(n_inputs, n_units) * 2 - 1
        self.output_layer = output_layer
        self.input, self.preactivation, self.activation = None, None, None

    def forward_step(self, input):
        self.input = input
        # print('sfsdfas', self.input.shape)
        self.preactivation = self.input @ self.weights + self.biases
        self.activation = relu(self.preactivation)
        return self.activation
    
    def backward_step(self, output, learning_rate):
        gradient_loss_over_activation = add_dimension_if_one(d_mse(output, self.activation) if self.output_layer else output)
        gradient_activation_over_preactivation = add_dimension_if_one(d_relu(self.preactivation))
        gradient_preactivation_over_weights = add_dimension_if_one(self.input)
        gradient_preactivation_over_bias = add_dimension_if_one(np.ones(self.preactivation.shape))
        gradient_preactivation_over_input = add_dimension_if_one(self.weights)
        gradient_loss_over_preactivation = gradient_activation_over_preactivation * gradient_loss_over_activation
        # print('\n Backward step:')
        # print('preactivation', self.preactivation.shape)
        # print('gradient_loss_over_activation', gradient_loss_over_activation.shape)
        # print('gradient_activation_over_preactivation', gradient_activation_over_preactivation.shape)
        # print('gradient_loss_over_preactivation', gradient_loss_over_preactivation.shape)
        # print('gradient_preactivation_over_weights', gradient_preactivation_over_weights.shape)
        # print('gradient_preactivation_over_bias', gradient_preactivation_over_bias.shape)
        # print('gradient_preactivation_over_input', gradient_preactivation_over_input.shape)
        gradient_loss_over_weights = gradient_preactivation_over_weights @ gradient_loss_over_preactivation.T
        gradient_loss_over_bias = gradient_preactivation_over_bias @ gradient_loss_over_preactivation.T
        gradient_loss_over_input = gradient_preactivation_over_input @ gradient_loss_over_preactivation

        self.weights -= gradient_loss_over_weights * learning_rate
        self.biases -= gradient_loss_over_bias[0] * learning_rate
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
    losses = []
    for i in range(epochs):
        for j in range(len(x)):
            y = mlp.forward_step(np.asarray([x[j]]))
            loss = mse(y, t[j])
            losses.append(loss)
            mlp.backpropagation(loss)
        print(f"Epoch: {i}, Loss: {losses[-1]}")


    y = np.ndarray(shape = x.shape)
    for i in range(x.shape[0]):
        y[i] = mlp.forward_step(np.expand_dims(np.asarray([x[i]]), axis = 0))


    plt.scatter(x,y)
    plt.show()


if __name__ == "__main__":
    main()


