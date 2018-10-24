import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
# for reproducibility

xy = np.loadtxt('data.csv', delimiter=',', dtype=np.float32)
# dataset
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# traing dataset
x_data_traing = x_data[0:int(len(x_data)*4/5)]
y_data_traing = y_data[0:int(len(x_data)*4/5)]

# testing dataset
x_data_testing = x_data[int(len(x_data)*4/5+1):int(len(x_data))]
y_data_testing = y_data[int(len(x_data)*4/5+1):int(len(x_data))]

print(len(x_data))


# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

cost_val = 0

# Traing set
for step in range(int(len(x_data)*4/5+1)):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data_traing, Y: y_data_traing})

print("Training Complete")

# Testing set
for step in range(int(len(x_data)/5+1)):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data_testing, Y: y_data_testing})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val, "\n")