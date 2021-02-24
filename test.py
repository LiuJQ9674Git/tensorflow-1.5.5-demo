import tensorflow as tf
import numpy as np

# tf.compat.v1.disable_eager_execution()
#
# x = tf.compat.v1.placeholder(tf.float32, shape=(1024, 1024))
# y = tf.matmul(x, x)
# with tf.compat.v1.Session() as sess:
#     rand_array = np.random.rand(1024, 1024)
#     print(sess.run(y, feed_dict={x: rand_array}))
#
# a = tf.compat.v1.placeholder("float")
# b = tf.compat.v1.placeholder("float")
#
# y = tf.matmul(a, b)
# sess = tf.compat.v1.Session()
# print( sess.run(y, feed_dict={a: 3, b: 3}))
# sess.close()

input1 = tf.placeholder("float")
input2 = tf.placeholder("float")
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# 输出:
# [array([ 14.], dtype=float32)]