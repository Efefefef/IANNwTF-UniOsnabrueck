import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt


def visualization(train_losses , train_accuracies , test_losses , test_accuracies):
    plt.figure()
    #testing takes place every 1875th (no of samples/batch size = 60000/32) steps. So we plot them there. 
    #we test once more than we train, so we need one data boint more
    xtest = np.arange(0,len(train_losses)+1875,1875)
    line1 , = plt.plot(train_losses , "b-")
    line2 , = plt.plot(xtest, test_losses , "r-") 
    line3 , = plt.plot(train_accuracies , "b:")
    line4 , = plt.plot(xtest, test_accuracies , "r:") 
    plt.xlabel("Training steps")
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

def test(model, image, target, loss_function):
    
    predictions = model(image)
    loss = loss_function(target, predictions)

    return loss, predictions

def train(model, train_ds, test_ds, epochs, loss_function, optimizer, train_losses, train_accuracies, test_losses, test_accuracies):
    
    for epoch in range(epochs):
        train_accuracy_aggregator = []
        test_accuracy_aggregator = []
        #perform a testing step
        #we do one testing step before training to see how good the network is by chance
        losses = []
        for image, target in test_ds:
            loss, test_predictions = test(model, image, target, loss_function)
            losses.append(loss)
            
            sample_test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(target, axis=1))
            test_accuracy_aggregator.append(sample_test_accuracy)
            
        #perform training
        losses = []
        for image, target in train_ds:
            loss, predictions = train_step(model, image, target, loss_function, optimizer)
            losses.append(loss)
            sample_train_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(target, axis=1))
            train_losses.append(tf.reduce_mean(losses))
            train_accuracy_aggregator.append(np.mean(sample_train_accuracy))
            train_accuracies.append(tf.reduce_mean(train_accuracy_aggregator))
       
        
        #only store mean of loss and accuracy for test steps
        test_losses.append(tf.reduce_mean(losses))         
        test_accuracies.append(tf.reduce_mean(test_accuracy_aggregator))


    #perform one extra test step because we can
    losses = []
    test_accuracy_aggregator = []

    for image, target in test_ds:
        loss, test_predictions = test(model, image, target, loss_function)
        losses.append(loss)
            
        sample_test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(target, axis=1))
        test_accuracy_aggregator.append(sample_test_accuracy)
    test_losses.append(tf.reduce_mean(losses))         
    test_accuracies.append(tf.reduce_mean(test_accuracy_aggregator))

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
