import tensorflow as tf
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist)
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
#我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：
x = tf.placeholder("float", [None, 784])

# 我们的模型也需要权重值和偏置量，TensorFlow有一个更好的方法来表示它们：Variable 。
# 一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
# 它们可以用于计算输入值，也可以在计算中被修改。
# 对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

#我们赋予tf.Variable不同的初值来创建不同的Variable：
# 在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
#计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值

y_ = tf.placeholder("float", [None,10])

# 计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 自动地使用反向传播算法(backpropagation algorithm)来有效地确定
# 你的变量是如何影响你想要最小化的那个成本值的。
# 然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。

# 们要求TensorFlow用梯度下降算法（gradient descent algorithm）
# 以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，
# TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
# 首先让我们找出那些预测正确的标签。tf.argmax 是一个非常有用的函数，
# 它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
# 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
# 比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#计算所学习到的模型在测试数据集上面的正确率。
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

