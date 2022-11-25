import datetime
import tensorflow_datasets as tfds 
import tensorflow as tf

# PREPARE THE DATA
def preprocess(data, task):
    batch_size = 32
    data = data.map(lambda image, target: (tf.cast(image, tf.float32) / 128. - 1, target))
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), data.shuffle(2000)))
    if task == 'larger_than_five':
        zipped_ds = zipped_ds.map(lambda x, y: (x[0], y[0], tf.cast((x[1]+y[1] > 4), tf.int32)))
        #  n-output=1, do not need one_hot
    else: # task == 'x-y'
        zipped_ds = zipped_ds.map(lambda x, y: (x[0], y[0], tf.cast((x[1]-y[1]), tf.int32)))
        zipped_ds = zipped_ds.map(lambda x, y, z: (x,y, tf.one_hot(z, 19)))
        # n-output=19, ranging from -9 to 9

    zipped_ds.cache()
    zipped_ds = zipped_ds.shuffle(2000)
    zipped_ds = zipped_ds.batch(batch_size)
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds

# CREATE MODEL FOR TWO TASKS
class FFN(tf.keras.Model):
    def __init__(self, optimiser, task):
        super().__init__()
        self.task = task
        self.optimizer = optimiser
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)

        # SPECIFIC CONSTRUCTOR FOR SUBTASK 1 (LARGER_THAN_FIVE)
        if self.task == 'larger_than_five':
            self.metrics_list = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Mean(name="loss")
            ]
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
            self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

        # SPECIFIC CONSTRUCTOR FOR SUBTASK 2 (X-Y)
        elif self.task == 'subtraction':   
            self.metrics_list = [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.Mean(name="loss")
            ]
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()
            self.out = tf.keras.layers.Dense(19, activation=tf.nn.softmax)
        

    @tf.function
    def __call__(self, images, training=False):
        x, y = images
        x = self.flatten(x)
        y = self.flatten(y)

        x = self.dense1(x)
        x = self.dense2(x)

        y = self.dense1(y)
        y = self.dense2(y)

        z = self.dense3(tf.concat([x, y], axis=1))
        z = self.out(z)
        return z

    # RESET ALL METRICS
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    # TRAIN STEP METHOD        
    def train_step(self, data):
        img1, img2, target = data

        with tf.GradientTape() as tape:
            prediction = self((img1, img2), training=True)
            loss = self.loss_function(target, prediction) 

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(loss)
        return {m.name: m.result() for m in self.metrics_list}
    
    # TEST STEP METHOD        
    def test_step(self, data):
        img1, img2, target = data
        prediction = self((img1, img2), training=False)
        loss = self.loss_function(target, prediction)
        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(loss)

# TRAINING LOOP
def training_loop(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path):
    for epoch in range(epochs):
        model.reset_metrics()

        for data in train_ds:
            model.train_step(data)

        with train_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)

        model.reset_metrics()
        
        for data in test_ds:
            model.test_step(data)

        with test_summary_writer.as_default():
            tf.summary.scalar(model.metrics[0].name, model.metrics[0].result(), step=epoch)
            tf.summary.scalar(model.metrics[1].name, model.metrics[1].result(), step=epoch)

        print("Epoch: {}, Loss: {}, Accuracy: {}".format(epoch, model.metrics[1].result(), model.metrics[0].result()))

    model.save_weights(save_path)

# TRAIN FUNCTION
def train(subtask, optimiser):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = f"models/{subtask}_{optimiser}/{current_time}"
    train_log_path = f"logs/{subtask}/{current_time}/train"
    test_log_path = f"logs/{subtask}/{current_time}/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    test_summary_writer = tf.summary.create_file_writer(test_log_path)

    epochs = 20

    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
    train_ds = preprocess(train_ds, subtask)
    test_ds = preprocess(test_ds, subtask)

    model = FFN(optimiser, subtask)

    training_loop(model, train_ds, test_ds, epochs, train_summary_writer, test_summary_writer, save_path)

# EXPERIMENTS

learning_rate = 0.001

# 1) SGD Optimizer (without momentum)
# momentum = 0.0
# optimiser1 = tf.keras.optimizers.experimental.SGD(learning_rate=learning_rate, momentum=momentum)

# 2) Adam Optimizer
optimiser2 = tf.keras.optimizers.Adam()

# BONUS POINT
# 3) SGD (with momentum =0.9)
# momentum = 0.9
# optimiser3 = tf.keras.optimizers.experimental.SGD(learning_rate= learning_rate, momentum=momentum)

#4) RMSProp
# optimiser4 = tf.keras.optimizers.experimental.RMSprop(learning_rate=learning_rate)

#5) AdaGrad
# optimiser5 = tf.keras.optimizers.experimental.Adagrad(learning_rate=learning_rate)

#train('larger_than_five', optimiser1)
train('subtraction', optimiser2)
