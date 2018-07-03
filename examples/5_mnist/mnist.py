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

W = tf.Variable(initial_value=tf.random_normal(
    [var_vector_length, nb_classes]))
b = tf.Variable(initial_value=tf.random_normal([nb_classes]))

"""
logit:
Logit is a function that maps probabilities [0, 1] to [-inf, +inf].

Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid. 
But Softmax also normalizes the sum of the values(output vector) to be 1.

Tensorflow "with logit": It means that you are applying a softmax function to logit numbers to normalize it. 
The input_vector/logit is not normalized and can scale from [-inf, inf].

This normalization is used for multiclass classification problems.
And for multilabel classification problems sigmoid normalization is used 
i.e. tf.nn.sigmoid_cross_entropy_with_logits

the vector of raw (non-normalized) predictions that a classification model generates, 
which is ordinarily then passed to a normalization function. 
If the model is solving a multi-class classification problem, 
logits typically become an input to the softmax function. 
The softmax function then generates a vector of (normalized) probabilities with one value for each possible class.
"""
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
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
