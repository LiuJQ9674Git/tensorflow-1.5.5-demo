import numpy
import tensorflow as tf

x = numpy.array([[1,2],
                 [3,4],
                 [5,6]])
print(x.shape)
print("x[1:2]:")
print(x[1:2]) #[[3 4]

print("x[0:1]:")
print(x[0:1]) #[[1 2]]

print("x[0:2]:")
print(x[0:2]) #[[1 2]
              # [3 4]]

print("x[:2]:")
print(x[:2]) # [[1 2]
             #  [3 4]]

print("x[:2,:1]")
print(x[:2,:1])
#[[1]
# [3]]

x = numpy.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])
print("x[1:]")
print(x[1:])
#[[4 5 6]
#[7 8 9]]

print("x[:1]")
print(x[:1])
# [[1 2 3]]

print("x[:,0]")
print(x[:,0])
#[1 4 7]

print("x[:,1]")
print(x[:,1])
#[2 5 8]
print("x[:,2]")
print(x[:,2])
#[3 6 9]
shape=[3, 3, 1, 32]

sess = tf.Session()

v1=tf.constant([[1.0,2.0],[3.0,4.0]])
t=tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]])
print("t shape:")
print(t.shape) #(3, 3, 1, 32)

sess = tf.Session()
print(sess.run(t))
# [[1. 2. 3.]
#  [4. 5. 6.]
#  [7. 8. 9.]]
sess.close()
