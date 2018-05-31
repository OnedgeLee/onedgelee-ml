import tensorflow as tf

# Build graph using TF operations

# placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# weight and bias type
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)

learning_rate = 0.01
gradient_w = tf.reduce_mean((W * X + b - Y) * X)
descent_w = W - learning_rate * gradient_w
update_w = W.assign(descent_w)
gradient_b = tf.reduce_mean(W * X + b - Y)
descent_b = b - learning_rate * gradient_b
update_b = b.assign(descent_b)

# launch the graph in a session
sess = tf.Session()

# initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# fit the line
for step in range(2001):
    cost_val, W_val, b_val, _, _ = sess.run([cost, W, b, update_w, update_b], feed_dict={
                                         X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
        