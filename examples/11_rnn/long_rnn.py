import tensorflow as tf
import numpy as np

sample = " if i want you"
idx2char = list(set(sample))
char2idx = {char: idx for idx, char in enumerate(idx2char)}

sample_idx = [char2idx[char] for char in sample]
x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]


# hyper parameters
sequence_length = len(sample) - 1
num_classes = len(idx2char)
dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
batch_size = 1

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

x_one_hot = tf.one_hot(X, num_classes)

cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
# batch, sequence, onehot의 3차원이기 때문에 axis=2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(500):
        loss_val, _ = sess.run([loss, train], feed_dict={X:x_data, Y:y_data})
        outputs_val, result = sess.run([outputs, prediction], feed_dict={X:x_data})

        if step % 50 == 0:
            result_str = [idx2char[idx] for idx in np.squeeze(result)]
            print(loss_val, ''.join(result_str))