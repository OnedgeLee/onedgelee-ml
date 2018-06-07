import tensorflow as tf

# set CSV file list
filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'], shuffle=False, name='filename_queue')

# set tensorflow reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# set initial
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]

xy = tf.decode_csv(value, record_defaults=record_defaults, field_delim=',')

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

predicted = tf.cast(hypothesis > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, predicted), tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 500 == 0:
        print(step, cost_val)

acc = 0
for step in range(760):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    _, _, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_batch, Y: y_batch})
    acc += a * 10

print(acc / 760)
coord.request_stop()
coord.join()