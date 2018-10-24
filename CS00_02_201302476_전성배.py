import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#Y = [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(4001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], Y: [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)