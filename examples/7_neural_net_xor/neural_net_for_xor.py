import tensorflow as tf
import numpy as np

"""
# logistic regression?

X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='variable')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='result')

W = tf.Variable(initial_value=tf.random_normal(shape=[2, 1]), name='weight')
b = tf.Variable(initial_value=tf.random_normal(shape=[1]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.sigmoid(logits)
# hypothesis = tf.div(1. / 1. + tf.exp(logits))

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)
                       * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
        if step % 50 == 0:
            acc_val = sess.run(accuracy, feed_dict={X: x_data, Y: y_data})
            print(step, cost_val, acc_val)
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print(h, p, a) 

# accuracy 75%가 최대 (OR GATE 혹은 NAND GATE)
"""
x_data = np.array(object=[[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array(object=[[0], [1], [1], [0]], dtype=np.float32)


X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='variable')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='result')

W1 = tf.Variable(initial_value=tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(initial_value=tf.random_normal([2]), name='bias1')

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(initial_value=tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(initial_value=tf.random_normal([1]), name='bias2')

hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)
predicted = tf.cast(hypothesis > 0.5, tf.float32)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)
                       * tf.log(1 - hypothesis))

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=0.3).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    print(sess.run([predicted, Y, accuracy], feed_dict={X: x_data, Y: y_data}))