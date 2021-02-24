import math
import tensorflow as tf

#输出是10个类别，因此我们需要定义一个参数去确定我们的输出是多少维度的：
NUM_CLASSES = 10
# 需要构建第一层的神经元了，我们知道第一层的神经元以图像作为输入，
# 因此我们首先需要去给出图像的大小的参数：
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

#尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求。
#函数会尽可能地构建图表，返回包含了预测结果（out prediction）的Tensor。
# 它以图片的占位符(image placeholder)作为输入，
# 借助ReLu（Rectified Linear Units）激活函数，构建一对全连接层。

def inference(images, hidden1_units, hidden2_units):
    #每一层都创建于一个唯一的tf.name_scope之下，创建于该作用域之下的所有元素都将带有其前缀。
    with tf.name_scope('hidden1'):
        #需要构建第一层的神经元了，我们知道第一层的神经元以图像作为输入，
        #因此我们首先需要去给出图像的大小的参数:IMAGE_PIXELS
        #我们有了这个参数之后，我们就可以构建输入层，也就是第一层的神经元的权重(weight)和偏置(bias)了，
        #第一层的输出为隐藏层第一层的输入：
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        #因此权重W的维度是：[IMAGE_PIXELS, hidden1_units]，
        # tf.truncated_normal初始函数将根据所得到的均值和标准差，生成一个随机分布。
        # 定义好了权重和偏置之后，接下来就需要将其和输入图像连接起来并经过激活函数：
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    #上面第一层神经元就构建好了，接下来我们以同样的方式构建第二层神经元：
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    #构建输出层
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

#往inference图表中添加生成损失（loss）所需要的操作（ops）
#函数通过添加所需的op来构建图
def loss(logits, labels):
    #首先，这个来自lables_placeholder是被编码成一个1-hot的向量。
    #例如：如果一个标签是被定义为数字“3”，那么它的表示形式如下所示：
    #[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    labels = tf.to_int64(labels)
    #使用tf.nn.sparse_softmax_cross_entropy_with_logits的交叉熵（cross-entropy）loss。
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

#往损失图表中添加计算并应用梯度（gradients）所需的操作。
def training(loss, learning_rate):
    #TensorBoard可以将训练过程中的各种绘制数据展示出来，
    # 包括标量(scallars)，图片(images)，音频(Audio)，计算图(graph)分布，
    # 直方图(histograms)和嵌入式向量。
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量，
# tf.nn.in_top_k(prediction, target, K):prediction就是表示你预测的结果，
# 大小就是预测样本的数量乘以输出的维度，类型是tf.float32等。
# target就是实际样本类别的标签（是mnist标签对应的下标值，
# 一个10维的标签代表一个值（这个十维的标签的最大值的下标），
# 具体的解析参考博客），大小就是样本数量的个数。
# K表示每个样本的预测结果的前K个最大的数里面是否含有target中的值。一般都是取1。
def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # print('labels',labels)
    return tf.reduce_sum(tf.cast(correct, tf.int32))