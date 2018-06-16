import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
          [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
          [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
# one-hot encoding


X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# hypothesis[None, 3]
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h = sess.run(hypothesis, feed_dict={X: x_data})
    print(h, sess.run(tf.argmax(h, 1)))
# tf.argmax의 두 번째 parameter은 one-hot-encoding을 적용할 차원

""" 
a3 = tf.Variable([[[0.1, 0.3, 0.5],
                   [0.3, 0.5, 0.1]],
                  [[0.5, 0.1, 0.3],
                   [0.1, 0.3, 0.5]],
                  [[0.3, 0.5, 0.1],
                   [0.5, 0.1, 0.3]]])

functions.showOperation(tf.argmax(a3, 0))
functions.showOperation(tf.argmax(a3, 1))
functions.showOperation(tf.argmax(a3, 2))

[출력 결과]
[[1 2 0]
 [2 0 1]]

[[1 1 0]
 [0 1 1]
 [1 0 1]]

[[2 1]
 [0 2]
 [1 0]] 

간단히, 0차원 진입, 1차원 진입, 2차원 진입 기준으로 동일 차원 기준으로 비교한다고 생각하면 됨
0: 0번 진입하여 최외각 원소들 3개 띄엄띄엄 비교
1: 1번 진입하여 2개 원소 비교
2: 끝까지 진입하여 3개 원소 비교
"""