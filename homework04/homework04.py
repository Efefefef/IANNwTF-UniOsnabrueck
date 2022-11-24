import datetime
import tensorflow_datasets as tfds 
import tensorflow as tf

# Prepare the data
def preprocess(data, task):
    batch_size = 32
    data = data.map(lambda image, target: (tf.cast(image, tf.float32) / 128. - 1, target))
    data = data.map(lambda image, target: (tf.reshape(image, (-1,)), target))
    data = data.map(lambda image, target: (image, tf.one_hot(target, 10)))
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), data.shuffle(2000)))
    zipped_ds = zipped_ds.map(lambda x, y: (x[0], y[0], tf.cast(x[1]==y[1], tf.int32)))
    zipped_ds.cache()
    zipped_ds = zipped_ds.shuffle(2000)
    zipped_ds = zipped_ds.batch(batch_size)
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds

class FFN(tf.keras.Model):
    def __init__(self, optimiser):
        super().__init__()
        self.metrics_list = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Mean(name="loss")
            ]
        self.optimizer = optimiser
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    @tf.function
    def __call__(self, images, training=False):
        x, y = images

        x = self.dense1(x)
        x = self.dense2(x)

        y = self.dense1(y)
        y = self.dense2(y)

        z = self.out(tf.concat([x, y], axis=1))
        return z

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
            
    def train_step(self, data):
        img1, img2, target = data

        with tf.GradientTape() as tape:
            prediction = self((img1, img2), training=True)
            loss = self.loss_function(target, prediction) 

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        img1, img2, target = data
        prediction = self((img1, img2), training=False)
        loss = self.loss_function(target, prediction)
        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(loss)


def training_loop(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path):
    for epoch in range(epochs):
        model.reset_metrics()

        for data in train_ds:
            model.train_step(data)

        with train_summary_writer.as_default():
            tf.summary.scalar('accuracy', model.metrics[0].result(), step=epoch)
            tf.summary.scalar('loss', model.metrics[1].result(), step=epoch)

        model.reset_metrics()
        
        for data in test_ds:
            model.test_step(data)

        with test_summary_writer.as_default():
            tf.summary.scalar('accuracy', model.metrics[0].result(), step=epoch)
            tf.summary.scalar('loss', model.metrics[1].result(), step=epoch)

        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, model.metrics[1].result(), model.metrics[0].result()))

    model.save_weights(save_path)

def train(subtask, optimiser):
    save_path = f"models/{subtask}_{optimiser}"
    train_log_path = f"logs/"
    test_log_path = f"logs/"
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    test_summary_writer = tf.summary.create_file_writer(test_log_path)
    epochs = 10

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
    train_ds = preprocess(train_ds, subtask)
    test_ds = preprocess(test_ds, subtask)

    model = FFN(optimiser)

    training_loop(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path)

train('larger_than_five', tf.keras.optimizers.Adam())