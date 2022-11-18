import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


def visualization(train_losses , train_accuracies , test_losses , test_accuracies):
    plt.figure()
    line1 , = plt.plot(train_losses , "b-")
    line2 , = plt.plot(test_losses , "r-") 
    line3 , = plt.plot(train_accuracies , "b:")
    line4 , = plt.plot(test_accuracies , "r:") 
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.show()


def prepare_mnist_data(mnist_data):
    mnist_data = mnist_data.map(lambda image, target: (tf.cast(image, tf.float32) / 128. - 1, target))
    mnist_data = mnist_data.map(lambda image, target: (tf.reshape(image, (-1,)), target))
    mnist_data = mnist_data.map(lambda image, target: (image, tf.one_hot(target, 10)))
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
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

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
    return loss, predictions

def test_step(model, image, target, loss_function):
    predictions = model(image)
    loss = loss_function(target, predictions)

    return loss, predictions

def train(model, train_ds, test_ds, epochs, loss_function, optimizer, train_losses, train_accuracies, test_losses, test_accuracies):

    losses = []
    test_accuracy_aggregator = []
    train_accuracy_aggregator = []

    for image, target in test_ds:
        loss, predictions = test_step(model, image, target, loss_function)
        losses.append(loss)
        test_accuracy_aggregator.append(np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1)))
    test_losses.append(tf.reduce_mean(losses))         
    test_accuracies.append(tf.reduce_mean(test_accuracy_aggregator))

    for image, target in train_ds:
        loss, predictions = test_step(model, image, target, loss_function)
        losses.append(loss)
        train_accuracy_aggregator.append(np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1)))
    train_losses.append(tf.reduce_mean(losses))         
    train_accuracies.append(tf.reduce_mean(train_accuracy_aggregator))

    print("Pretraining, Loss: {}, Accuracy: {}".format(train_losses[-1], train_accuracies[-1])) 
    print("Pretraining, Loss: {}, Accuracy: {}".format(test_losses[-1], test_accuracies[-1])) 
    
    #perform testing
    for epoch in range(epochs):
        train_accuracy_aggregator = []
        test_accuracy_aggregator = []
        losses = []
        for image, target in test_ds:
            loss, predictions = test_step(model, image, target, loss_function)
            losses.append(loss)
            test_accuracy_aggregator.append(np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1)))

        test_losses.append(tf.reduce_mean(losses))         
        test_accuracies.append(tf.reduce_mean(test_accuracy_aggregator))

        #perform training
        losses = []
        for image, target in train_ds:
            loss, predictions = train_step(model, image, target, loss_function, optimizer)
            losses.append(loss)
            train_accuracy_aggregator.append(np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1)))
            
        train_losses.append(tf.reduce_mean(losses))
        train_accuracies.append(tf.reduce_mean(train_accuracy_aggregator))
       
        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch + 1, train_losses[-1], train_accuracies[-1])) 
        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch + 1, test_losses[-1], test_accuracies[-1])) 

    return train_losses, train_accuracies, test_losses, test_accuracies


num_epochs = 10
learning_rate = 0.1

model = MyModel()
(train_ds, test_ds), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
)

train_losses , train_accuracies , test_losses , test_accuracies = [], [], [], []

train_losses, train_accuracies, test_losses ,test_accuracies = train(
    model = model, 
    train_ds= prepare_mnist_data(train_ds), 
    test_ds= prepare_mnist_data(test_ds), 
    epochs= num_epochs, 
    loss_function= tf.keras.losses.CategoricalCrossentropy(), 
    optimizer = tf.keras.optimizers.SGD(learning_rate),
    train_losses=train_losses,
    train_accuracies=train_accuracies,
    test_losses=test_losses,
    test_accuracies=test_accuracies
)

visualization(train_losses, train_accuracies, test_losses, test_accuracies)
