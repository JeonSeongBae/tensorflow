import tensorflow as tf

W = tf.Variable([0.])

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(10):
    cost_val, W_val, _ = sess.run([cost, W, train], feed_dict={X: [1,2,3,4,5], Y: [1,3,1,3,5]})
    print(step, cost_val, W_val)