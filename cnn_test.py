import tensorflow as tf
import numpy as np
import input_data

batch_size = 128
test_size = 256

img_size = 28
num_classes = 10

# 为输入图像定义占位符变量X。该张量的数据类型被设置为float32，
# 形状设置为[None, img_size, img_size, 1]。此处None表示该张量可以保存任意数量的图像:
X = tf.placeholder("float", [None, img_size, img_size, 1])
# 该占位符变量的形状为[None, num_classes]，代表该张量可以保存任意数量的标签，
# 且每个标签为长度num_classes的一个向量。
Y = tf.placeholder("float", [None, num_classes])

mnist = input_data.read_data_sets("data/",one_hot=True)
trX, trY, teX, teY = mnist.train.images, \
                     mnist.train.labels, \
                     mnist.test.images, \
                     mnist.test.labels
def printShape():
    print("原始数据的类型")
    print(trX.shape) #(55000, 784)
    print(trY.shape) #(55000,)
    print(teX.shape) #(10000, 784)
    print(teY.shape) #(10000,)
printShape()

# trX和trY图像集必须根据输入形状进行变形:
trX = trX.reshape(-1, img_size, img_size, 1)
teX = teX.reshape(-1, img_size, img_size, 1)

def printShapeX():
    print("转型数据的类型")
    print(trX.shape) #(55000, 28, 28, 1)
    print(teX.shape) #(10000, 28, 28, 1)
printShapeX()

# 定义网络权重
# 函数能构建给定形状的新变量，并初始化网络权重为随机值
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# 第一个卷积层中的每个神经元均由输入张量的一个小子集卷积而来，其维度为3 × 3 × 1。
# 数值32是这一层的特征图数量。因此，可以将权重w定义如下:
w = init_weights([3, 3, 1, 32])

# 第二个卷积层中的每个神经元均由第一个卷积层中3 × 3 × 32个神经元卷积而来。
# 输入的数量上升至32，
# 数值64代表该层获得的输出特征数量。
# 权重w2定义如下:
w2 = init_weights([3, 3, 32, 64])

# 第三个卷积层由前一层的3 × 3 × 64个神经元卷积而来，
# 其输出特征数量为128:
w3 = init_weights([3, 3, 64, 128])

# 第四层是一个全连接层。该层接收128 × 4 × 4的输入，输出数量为625:
w4 = init_weights([128 * 4 * 4, 625])

#输出层接收625个输入，其输出为类的数量:
w_o = init_weights([625, num_classes])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    conv1 = tf.nn.conv2d(X, w, strides=[1, 1, 1, 1],
                         padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                           strides = [1, 2, 2, 1],
                           padding = 'SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    conv2 = tf.nn.conv2d(conv1, w2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3 = tf.nn.conv2d(conv2, w3,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    conv3 = tf.nn.relu(conv3)
    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')
    FC_layer = tf.reshape(FC_layer,[-1,w4.get_shape().as_list()[0]])
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)

    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)

cost = tf.reduce_mean(Y_)

optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

predict_op = tf.argmax(py_x, 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size), \
                             range(batch_size, \
                                   len(trX)+1, \
                                   batch_size))
        for start, end in training_batch:
             print(start)
             sess.run(optimizer, feed_dict={X: trX[start:end], \
                                            Y: trY[start:end],\
                                            p_keep_conv: 0.8,\
                                            p_keep_hidden: 0.5})

