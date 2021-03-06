import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-03-diabetes.csv', dtype=np.float32, delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(10001):
    cost_val, _ = sess.run([cost, train],feed_dict={X: x_data, Y: y_data})
    if step % 200:
     print(step, cost_val)

h, p, a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: x_data, Y:y_data})
print(h, p, a)

# decode csv로 연습해보기
