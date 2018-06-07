import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', dtype=np.float32, delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.float32, shape=[None, 1])

nb_classes = 7
Y_one_hot = tf.one_hot(Y, nb_classes)
# Y_one_hot[?, 1, 7]
Y_one_hot = tf.reshape(Y_one_hot, shape=[-1, nb_classes])
# Y_one_hot[?, 7]

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

