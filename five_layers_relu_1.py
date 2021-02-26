import input_data
import tensorflow as tf
import math


logs_path = 'log_simple_stats_5_layers_relu_softmax'
# 配置参数
batch_size = 100
learning_rate = 0.5
training_epochs = 10

max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000
# 数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# 输入层开始构建网络架构
# 输入层现在是一个形状为[1×784]的张量，代表待分类图像
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

XT = tf.placeholder(tf.float32, [None, 28, 28, 1])

lr = tf.placeholder(tf.float32)


L = 200
M = 100
N = 60
O = 30

# 网络的第一层接收待分类输入图像的像素，与W1权重连接组合，
# 并与对应的B1偏差张量相加:
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  
B1 = tf.Variable(tf.ones([L])/10)
# 第一层通过sigmoid激活函数，将自己的输出传给第二层:
XX = tf.reshape(X, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)

W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)

W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)

W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)

W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
#ReLU的运算速 度很快，因为它不需要进行指数运算，这一点与sigmoid和tanh等激活函数不同。
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100


correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
#均方法差
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

tf.summary.scalar("cost", cross_entropy)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logs_path, \
                                    graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        batch_count = int(mnist.train.num_examples/batch_size)
        for i in range(batch_count):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # 读取数据为(10000, 784),需要转换为(?, 28, 28, 1)
            batch_x = batch_x.reshape(-1, 28, 28, 1)

            learning_rate = min_learning_rate+\
                            (max_learning_rate - min_learning_rate)\
                            * math.exp(-i/decay_speed)
            _, summary = sess.run([train_step, summary_op],\
                                  {X: batch_x, Y_: batch_y,\
                                   lr: learning_rate})
            writer.add_summary(summary,\
                               epoch * batch_count + i)
        #if epoch % 2 == 0:
        print("Epoch: ", epoch)
    # 读取数据为(10000, 784),需要转换为(?, 28, 28, 1)
    XT=mnist.test.images
    XT=XT.reshape(-1, 28, 28, 1)
    # Accuracy:  0.9753
    print("Accuracy: ", accuracy.eval(feed_dict={X: XT, Y_: mnist.test.labels}))
    print("done")

