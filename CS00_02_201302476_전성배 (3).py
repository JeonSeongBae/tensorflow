import tensorflow as tf
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [2.2, 5.2, 6.1, 7.9, 10.5, 11.8, 15, 16, 18.2, 20]

W = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Our hypothesis XW+b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session
sess = tf.Session()
#Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

# Fit the line
for i in range(0, 40):
    feed_W = i * 0.1
    curr_cost, curr_W, curr_b = sess.run([cost, W, b], feed_dict={W: feed_W, b: 0.5})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# print
print(cost_val[tf.argmin(cost_val).eval(session=sess)])
print(W_val[tf.argmin(cost_val).eval(session=sess)])

# Show the cost function
plt.plot(W_val, cost_val)
plt.show()