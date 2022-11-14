# Homework 03 - Due 17.11.2022 - 23.59
# MNIST Classification

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

train_ds, test_ds = tfds.load('mnist',
                              split=['train', 'test'],
                              as_supervised= True)
# PREPARING DATA (MNIST)
def prepare_data (mnist):
    # flatten
    mnist = mnist.map(lambda img, trg: (tf.reshape(img, (-1,)), trg))
    # convert uint8 --> float32
    mnist = mnist.map(lambda img, trg: (tf.cast(img, tf.float32), trg))
    # normalization from [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, trg: ((img / 128.) - 1., trg))
    #  one-hot targets
    mnist = mnist.map(lambda img, trg: (img, tf.one_hot(trg, depth=10)))

    mnist = mnist.cache()
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(32)

    return mnist

# BUILDING NETWORK
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(256, activation=tf.nn.relu)
        self.dense2 = Dense(256, activation=tf.nn.relu)
        self.out = Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x

# FUNCTION FOR TRAINING AND TESTING LOOP
def training(model, input, target, loss_function, optimizer):
    # loss_function & optimizer : instances of respective tf classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_data, loss_function):
    test_acc_aggregator = []
    test_loss_aggregator = []

    for (inp, target) in test_data:
        prediction = model(inp)
        sample_test_loss = loss_function(target, prediction)
        sample_test_acc = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_acc = np.mean(sample_test_acc)
        test_loss_aggregator.append(sample_test_loss)
        test_acc_aggregator.append(sample_test_acc)

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_acc = tf.reduce_mean(test_acc_aggregator)

    return test_loss, test_acc

def visualization(train_loss, test_loss, test_acc):
    """
    Visualizing accuracy and loss for training and test data
    using the mean of each epoch.
    Legend:
        Loss= regular line
        Accuracy= dotted line
        Training= blue line
        Test = red line

    Parameters:
         train_loss: numpy.ndarray = training losses
         train_acc: numpy.ndarray = training accuracies
         test_loss: numpy.ndarray = test losses
         test_acc: numpy.ndarray = test accuracies
    """
    plt.figure()
    plt.plot(train_loss, "b-", label="Training Loss")
    plt.plot(test_loss, "r-", label="Test Loss")
    #plt.plot(train_acc, "b:", label="Training Accuracy")
    plt.plot(test_acc, "r:", label="Test Accuracy")
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

# PREPARING DATA
train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)


# FOR CHECKING THE CODE, UNCOMMENT THIS CODE
train_dataset = train_dataset.take(10000)
test_dataset = test_dataset.take(1000)

# INITIALISING HYPERPARAMETERS
n_epochs= 10
learn_rate = 0.1

# initiating model
model = MyModel()
# categorical entropy loss
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# optimised SGD
optimizer = tf.keras.optimizers.SGD(learn_rate)

# lists for visualisation
train_losses = []
test_losses = []
test_accuracies = []
train_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# TRAINING OVER N-EPOCHS
for epoch in range(n_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = training(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing
    test_loss, test_acc = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)


visualization(train_loss=train_losses, test_loss=test_losses, test_acc=test_accuracies)