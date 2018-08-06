import tensorflow as tf
import numpy as np
import pprint as pp

# hihello 학습
batch_size = 1
sequence_length = 6
num_classes = 5

idx2char = ['h', 'i', 'e', 'l', 'o']
input_set = [0, 1, 2, 3, 4]

for i in range(len(input_set)):
    zero_arr = np.zeros(shape=num_classes, dtype=np.float)
    zero_arr[input_set[i]] = 1
    input_set[i] = zero_arr

[h, i, e, l, o] = input_set

input_shape = [batch_size, sequence_length, num_classes]
hidden_size = 5
hidden_size = 5

x_data = [[h, i, h, e, l, l]]
y_data = [[1, 0, 2, 3, 3, 4]]
X = tf.placeholder(dtype=tf.float32, shape=[
                   None, sequence_length, num_classes])
Y = tf.placeholder(dtype=tf.int32, shape=[
                   None, sequence_length])

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones(shape=[batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(step, l, result, y_data)
