import tensorflow as tf

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
d = a + b
c = a * b
e = c + d

sess = tf.Session()
print(sess.run(e, feed_dict={a: 5, b: 3}))
