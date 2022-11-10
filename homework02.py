import numpy as np
from matplotlib import pyplot as plt

# Create data set
def create_dataset():
    x = np.random.rand(100)
    t = x**2
    x = [x.reshape(1, 1) for x in x]
    t = [t.reshape(1, 1) for t in t]

    # Plot to show how the function is supposed to look like
    plt.scatter(x, t, s=5)
    plt.title("The function to be approximated")
    plt.ylabel("target")
    plt.show()
    return x, t

# Shuffles dataset
def shuffle(x,t):
    np.random.shuffle(x)
    np.random.shuffle(t)
    return x, t

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
    def __init__(self, n_inputs, n_units):
        self.biases = np.zeros((n_units, 1))
        self.weights = np.random.rand(n_inputs, n_units)
        self.n_units = n_units
        self.input, self.preactivation, self.activation = None, None, None

    def forward_step(self, input):
        self.input = input
        self.preactivation = self.weights.T @ self.input + self.biases
        self.activation = relu(self.preactivation)
        return self.activation
    
    def backward_step(self, output, learning_rate):
        gradient_loss_over_activation = output
        gradient_activation_over_preactivation = d_relu(self.preactivation)
        gradient_preactivation_over_weights = self.input
        gradient_preactivation_over_bias = np.ones((self.n_units, 1))
        gradient_preactivation_over_input = self.weights

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

    def train(self, x, t):
        losses = []
        for i in range(len(x)):
            y = self.forward_step(x[i])
            loss = mse(y, t[i])
            self.backpropagation(d_mse(y, t[i]))
            losses.append(loss)
        return losses


def main():
    epoch_size = 100
    x, t = create_dataset()
    mlp = MLP([Layer(1, 10), Layer(10, 10), Layer(10, 1)], learning_rate=0.01)
    mean_loss = []
    losses = []
    for e in range(epoch_size):
        x,t = shuffle(x,t)
        losses_per_epoch = mlp.train(x, t)
        losses.append(losses_per_epoch)
        mean_loss.append(np.mean(losses))

    # Plot the mean loss per epoch
    plt.plot(range(epoch_size), mean_loss)
    plt.title("Mean loss per Epoch")
    plt.xlabel("epoch")
    plt.ylabel("mean loss")
    plt.show()

if __name__ == "__main__":
    main()


