from random import randint
import numpy as np
import tensorflow as tf
import input_data
logs_path = 'log_simple_stats_softmax'

batch_size = 100
learning_rate = 0.5
training_epochs = 10

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 定义网络模型
# 输入网络包含从MNIST数据集抽取的一系列图像，每个图像的大小为28 × 28像素:
X = tf.placeholder(tf.float32, [None, 28, 28, 1],name="input")

# 需要解决的问题是如何为每个分类(数字0~9)分别指定一个概率值。
# 对应的输出描述了一个概率分布，我们可以从中得到对待检验值的一个预测。
# 输出网络保存在由10个元素张量组成的 下列占位符中:
Y_ = tf.placeholder(tf.float32, [None, 10])

# 权重同时考虑了隐藏层的大小(10个神经元)和输入规模。
# 权重的值在每一轮迭代过程中必 须更新, 权重矩阵为W[784, 10]，其中784 = 28 × 28。
W = tf.Variable(tf.zeros([784, 10]))

# 将图像展开为一条一维像素线;形状定义中的数字-1代表保存元素数量的唯一可能的维度。
XX = tf.reshape(X, [-1, 784])

# 为网络定义偏差bias，它表示触发信号相对于原始输入信号的平移量。形式上，
# 偏差扮演的角色和权重没什么差别，都用于调节发射/接收信号的密度。
# 其大小(= 10)等于隐藏层神经元数的总和。
b = tf.Variable(tf.zeros([10]))

# input、weight和bias张量的大小得到合理定义之后，便可定义evidence参数，用来量化
# 一个图像是否属于某个特定的类。
# 这里的神经网络只有一个隐藏层，由10个神经元组成。由前馈网络的定义可知，
# 同一层的所 有神经元都有相同的激活函数。
evidence = tf.matmul(XX, W) + b

# 在我们的模型里，激活函数是softmax函数，它将evidence参数转化为图像可能属于的10个类的概率。
# 输出矩阵Y由100行10列组成。
Y = tf.nn.softmax(evidence,name="output")

# 为训练模型并判断该模型性能，必须定义一个度量单位。
# 实际上，接下来的步骤就是获取W和b的值，使该度量单位的值最小，并评价该模型的好坏。
# 有很多度量单位可以计算实际输出和期望输出之间的误差程度。
# 最常见的误差分数是均方差，但一些研究也针对这类网络提出了其他类似的度量单位。
# cross_entropy(交叉熵)误差函数。定义如下
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0

# 梯度下降算法，将该误差函数最小化
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)

# 若网络输出值Y与期望输出值Y_相等，则说明预测正确:
correct_prediction = tf.equal(tf.argmax(Y, 1),tf.argmax(Y_, 1))
# correct_prediction变量可用于定义模型的准确率。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 接下来定义汇总
tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

# 模型建立session会话，用以触发训练和测试步骤

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # 网络的训练过程是迭代的。在每轮学习(或每个时期)中，
    # 网络会使用选定的子集(或批量集)对突触权重进行小型更新。
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 子集会被feed_dict语句调用，用以在训练过程中对网络进行馈给。
            # 在每轮学习中: 修改权重以最小化误差函数
            #             并writer.add_summary语句将结果添加到汇总
            batch_x_re=batch_x.reshape(-1, 28, 28, 1)
            _, summary = sess.run([train_step, summary_op],
                                  feed_dict={X: batch_x_re,
                                             Y_: batch_y})
            writer.add_summary(summary, epoch * batch_count + i)
        print("Epoch: ", epoch)

    test=mnist.test.images
    test_re=test.reshape(-1, 28,28,1)
    test_labels=mnist.test.labels
    #  0.9219
    print("Accuracy: ", accuracy.eval(
        feed_dict={X: test_re,
        Y_: mnist.test.labels}))

    #网络测试完成后，继续留在会话内部，在单一图像上运行网络模型。
    # 例如，可以利用randint函数，从mnist.test数据库随机选取一张图片。
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]
    img=img.reshape(28,28,1) #一张图片
    # 这样即可将前面实现的分类器用在选定的图像上。
    # sess.run函数的参数分别为网络的输出和输入。
    # tf.argmax(Y, 1)函数返回Y张量的最大索引值，即我们要找的图像。
    # feed_dict={X: [img]}，允许我们将选定的图像馈给网络。
    classification = sess.run(tf.argmax(Y, 1), feed_dict={X: [img]})
    print('Neural Network predicted', classification[0])
    print('Real label is:', np.argmax(mnist.test.labels[num]))

    #保存数据
    saver = tf.train.Saver()
    save_path = saver.save(sess, "softmax_mnist")
    print("Model saved to %s" % save_path)

