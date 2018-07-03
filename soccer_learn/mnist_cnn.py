import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# reproducibility
tf.set_random_seed(777)

mnist = input_data.read_data_sets('MNIST_data/', one_hot='True')

# hyper parameters
learning_rate = 1e-3
training_epochs = 20
batch_size = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):

            # flag for dropout
            self.training = tf.placeholder(tf.bool)

            # input placeholder
            self.X = tf.placeholder(tf.float32, [None, 784])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # input layer
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])

            # convolutional / pooling layer #1
            conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='SAME')
            dropout1 = tf.layers.dropout(inputs=pool1, rate=0.3, training=self.training)

            # convolutional / pooling layer #2
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
            dropout2 = tf.layers.dropout(inputs=pool2, rate=0.3, training=self.training)

            # convolutional / pooling layer #3
            conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, padding='SAME')
            dropout3 = tf.layers.dropout(inputs=pool3, rate=0.3, training=self.training)

            # dense layer with relu
            flat = tf.reshape(dropout3, [-1, 4 * 4 * 128])
            dense4 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
            dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=self.training)

            # logits layer
            self.logits = tf.layers.dense(inputs=dropout4, units=10)

        # define cost/loss & optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})
    
    def g