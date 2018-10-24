import tensorflow as tf

# a와 b를 tf.constant [1,2,3]와 [3,2,1]으로 구현
a = tf.constant([1,2,3])
b = tf.constant([3,2,1])

mul1 = a * b
mul2 = a * b
mul3 = a * b

add1 = mul2 + mul3
add2 = mul1 + mul3
add3 = mul1 + mul2

add4 = add1 + add2
add5 = add2 + add3

add6 = add4 + add5

sess = tf.Session()
print(sess.run(add6))

#a와 b를 tf.placeholder를 이용하여 구현
a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)

mul1 = a * b
mul2 = a * b
mul3 = a * b

add1 = mul2 + mul3
add2 = mul1 + mul3
add3 = mul1 + mul2

add4 = add1 + add2
add5 = add2 + add3

add6 = add4 + add5

sess = tf.Session()
print(sess.run(add6, feed_dict={ a : [[1,5,6],[3,4,5]], b : [[5,4,3],[3,5,6]]}))
