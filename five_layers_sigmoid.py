import tensorflow as tf
import input_data
import math
#
logs_path = 'log_simple_stats_5_layers_sigmoid'
batch_size = 100
learning_rate = 0.003
training_epochs = 100 #10 0.9442 30 0.9743 0.9781

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

L = 200
M = 100
N = 60
O = 30


# 构建网络架构

# 输入层现在是一个形状为[1×784]的张量，代表待分类图像
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
XX = tf.reshape(X, [-1, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])
# 网络的第一层接收待分类输入图像的像素，与W1权重连接组合，并与对应的B1偏差张量相加
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1)) #L=200
B1 = tf.Variable(tf.zeros([L]))
# 第一层通过sigmoid激活函数，将自己的输出传给第二层:
# 向前传播算法
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)

# 第二层接收第一层的输出Y1，并将其与W2权重连接组合，再加上对应的B2偏差张量:
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1)) #M=100
B2 = tf.Variable(tf.zeros([M]))
# 第二层通过sigmoid激活函数，将输出传给第三层:
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)

# 第三层接收第二层的输出Y2，与W3权重连接组合起来，并与对应的B3偏差张量相加:
W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))  #N=60
B3 = tf.Variable(tf.zeros([N]))
# 第三层通过sigmoid激活函数，将输出传给第四层:
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3) #矩阵乘法

# 第四层接收第三层的输出Y3，与W4权重连接组合起来，并与对应的B4偏差张量相加:
W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1)) #O=30
B4 = tf.Variable(tf.zeros([O]))
# 第四层通过sigmoid激活函数，将输出传给第五层
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)

# 第五层将从第四层接收O = 30的激励作为输入，这些输入会通过softmax激活函数，转化为 每个数字对应的概率:
W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
# 向前传播
Ylogits = tf.matmul(Y4, W5) + B5

# 损失函数
Y = tf.nn.softmax(Ylogits)

# 损失函数为，目标与softmax激活函数产生的结果之间的交叉熵cross-entropy
# 交差熵计算
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# tf.train.AdamOptimizer使用Kingma和Ba’s Adam算法控制学习率
# AdamOptimizer比简单的tf.train.GradientDescentOptimizer有几个优势，
# 实际上，前者使用了更大的有效更新步长，这样算法不需要经过微调即可收敛。
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x=batch_x.reshape(-1,28, 28, 1)
            _, summary = sess.run([train_step, summary_op],\
                                  feed_dict={X: batch_x,\
                                             Y_: batch_y})
            writer.add_summary(summary,\
                               epoch * batch_count + i)
        print("Epoch: ", epoch)
    test=mnist.test.images
    # 转型
    test=test.reshape(-1, 28,28,1) #<class 'tuple'>: (10000, 28, 28, 1)
    print("Accuracy: ", accuracy.eval(feed_dict={X: test, Y_: mnist.test.labels}))
    print("done")

