import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import random
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

nb_classes = 10
var_vector_length = 784

X = tf.placeholder(dtype=tf.float32, shape=[
                   None, var_vector_length], name='variable')
Y = tf.placeholder(dtype=tf.float32, shape=[None, nb_classes], name='result')

W1 = tf.get_variable('W1', shape=[var_vector_length, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))

W2 = tf.get_variable('W2', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))

W3 = tf.get_variable('W3', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([256]))

W4 = tf.get_variable('W4', shape=[256,256], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))

W5 = tf.get_variable('W5', shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]))

keep_prob = tf.placeholder(tf.float32)

logits1 = tf.matmul(X, W1) + b1
layer1 = tf.nn.relu(logits1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

logits2 = tf.matmul(layer1, W2) + b2
layer2 = tf.nn.relu(logits2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

logits3 = tf.matmul(layer2, W3) + b3
layer3 = tf.nn.relu(logits3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

logits4 = tf.matmul(layer3, W4) + b4
layer4 = tf.nn.relu(logits4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

logits5 = tf.matmul(layer4, W5) + b5
hypothesis = tf.nn.softmax(logits5)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits5, labels=Y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 20
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))
    
    print('Accuracy:', accuracy.eval(session=sess,  feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
