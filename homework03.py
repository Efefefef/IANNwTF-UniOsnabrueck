import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib as plt

(train_ds, test_ds), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
)

def prepare_mnist_data(mnist_data):
    mnist_data = mnist_data.map(lambda image, target: (tf.cast(image, tf.float32) / 128. - 1, target))
    mnist_data = mnist_data.map(lambda image, target: (tf.reshape(image, (-1,)), target))
    mnist_data = mnist_data.map(lambda image, target: (tf.one_hot(target, 10), image))
    mnist_data.cache()
    mnist_data = mnist_data.shuffle(1000)
    mnist_data = mnist_data.batch(32)
    mnist_data = mnist_data.prefetch(20)
    return mnist_data

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layeres.Dense(10, activation=tf.nn.softmax)

        @tf.function
        def call(self, inputs):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.out(x)
            return x

def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(input)
        loss = loss_function(target, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_ds, target, loss_function):
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (image, target) in test_ds:
        predictions = model(image)
        sample_test_loss = loss_function(target, predictions)
        test_loss_aggregator.append(sample_test_loss.numpy())
        sample_test_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1))
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))
    
    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy

def train(model, train_ds, test_ds, epochs, loss_function, optimizer, train_losses, train_accuracies, test_losses, test_accuracies):
    for epoch in range(epochs):
        losses = []
        for image, target in train_ds:
            loss = train_step(model, image, target, loss_function, optimizer)
            losses.append(loss)
        train_losses.append(tf.reduce_mean(losses))

    test(test_ds, test_losses, test_accuracies)
    return train_losses, train_accuracies, test_losses, test_accuracies


num_epochs = 10
learning_rate = 0.1

model = MyModel()

train_losses , train_accuracies , test_losses , test_accuracies = [], [], [], []

train_losses, train_accuracies, test_losses ,test_accuracies = train(
    model, prepare_mnist_data(train_ds), prepare_mnist_data(test_ds), num_epochs, tf.keras.losses.CategoricalCrossentropy(), tf.keras.optimizers.SGD(learning_rate)
)

def visualization(train_losses , train_accuracies , test_losses , test_accuracies):
    plt.figure()
    line1 , = plt.plot(train_losses , "b-")
    line2 , = plt.plot(test_losses , "r-") 
    line3 , = plt.plot(train_accuracies , "b:")
    line4 , = plt.plot(test_accuracies , "r:") 
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.show()