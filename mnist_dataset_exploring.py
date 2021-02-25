import tensorflow as tf
from matplotlib import pyplot as plt
import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
__input=mnist

def printMNIST():
    print(mnist)
    print(mnist.train.images.shape) #(55000, 784)
    #转型
    image_re=mnist.train.images
    image_re=image_re.reshape(-1, 28, 28, 1)
    print(image_re.shape) #(55000, 28, 28, 1)
    print(mnist.train.labels.shape) #(55000, 10)
    print(mnist.test.images.shape) #(10000, 784)
    print(mnist.test.labels.shape) #(10000, 10)

    print(type(mnist))
    print(mnist.train.num_examples)#55000
    print(mnist.test.num_examples)#10000

    img_train = mnist.train.images
    label_train = mnist.train.labels

    img_test = mnist.test.images
    label_test = mnist.test.labels

    print(type(img_train))#<class 'numpy.ndarray'>
    print(type(label_train))#<class 'numpy.ndarray'>
    print(type(img_test))#<class 'numpy.ndarray'>
    print(type(label_test))#<class 'numpy.ndarray'>
    print(img_train.shape)#(55000, 784) 28*28的图片
    print(label_train.shape)#(55000, 10)
    print(img_test.shape)#(10000, 784)
    print(label_test.shape)#(10000, 10) #one hot coding便于取最大概率
printMNIST()
image_0 = __input.train.images[1]

print(image_0.shape) #(784,)
image_0 = np.resize(image_0,(28,28))
print(image_0.shape) #(28, 28)
label_0 = __input.train.labels[1]
# [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
print(label_0)
plt.imshow(image_0, cmap='Greys_r')
plt.show()