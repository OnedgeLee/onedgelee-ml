import tensorflow as tf
import random
import numpy as np


nb_classes = 2
var_vector_length = 1

xy = np.loadtxt('b.csv', dtype=np.float32, delimiter=',')
x_train_data = xy[20:50, [0]]
y_train_data = xy[20:50, [1]]

# y_train_one_hot = tf.one_hot(y_train_data, nb_classes)
# y_train_one_hot = tf.reshape(y_train_one_hot, shape=[-1, nb_classes])

x_test_data = xy[:10, [0]]
y_test_data = xy[:10, [1]]

# y_test_one_hot = tf.one_hot(y_test_data, nb_classes)
# y_test_one_hot = tf.reshape(y_test_one_hot, shape=[-1, nb_classes])

X = tf.placeholder(dtype=tf.float32, shape=[
                   None, var_vector_length], name='variable')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='result')

Y_o = tf.one_hot(tf.cast(Y,tf.int32), nb_classes)
Y_o = tf.reshape(Y_o, shape=[-1, nb_classes])

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

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits5, labels=Y_o)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_o, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        hypo_val, cost_val, _ = sess.run([hypothesis, cost, optimizer], feed_dict={X: x_train_data, Y: y_train_data, keep_prob:0.7})
        if step % 200 == 0:
            acc_val = sess.run(accuracy, feed_dict={X: x_test_data, Y: y_test_data, keep_prob:1})
            print(step, hypo_val, cost_val, acc_val)
