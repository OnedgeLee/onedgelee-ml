import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 1000])
Y = tf.placeholder(tf.float32, shape=[None, 2])

X_img = tf.reshape(X, [-1, 28, 28, 1])
F1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# 왜 편차를 0.01로 잡았는가? 생각해 볼 것

L1 = tf.nn.conv2d(X_img, F1, strides=[1, 1, 1, 1], padding='SAME')
# [?, 28, 28, 32]
L1 = tf.nn.relu(L1)
# [?, 28, 28, 32]
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
# [?, 14, 14, 32]
# batch와 in_channels는 특별한 이유가 없을 경우 1로 주는게 타당

F2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
L2 = tf.nn.conv2d(L1, F2, strides=[1, 1, 1, 1], padding='SAME')
# [?, 14, 14, 64]
L2 = tf.nn.relu(L2)
# [?, 14, 14, 64]
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
# [?, 7, 7, 64]
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])

W3 = tf.get_variable('W3', shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 5
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    
    print('Accuracy:', accuracy.eval(session=sess,  feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
