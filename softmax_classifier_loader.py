import matplotlib.pyplot as plt
import tensorflow as tf
import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()
# 导入已保存的计算图元数据，其中包含我们所需模型的所有拓扑结构及相应变量:
new_saver = tf.train.import_meta_graph('softmax_mnist.meta')
# 然后导入检验点文件，其中包含训练过程中得出的权重值:
new_saver.restore(sess, 'softmax_mnist')

# 若要运行已载入的模型，我们需要其计算图。可以通过以下函数调用:
tf.get_default_graph()
# 下面的函数将返回当前线程中所用的默认图:
tf.get_default_graph().as_graph_def()

# 定义x和y_conv变量，并将它们和我们需要处理的节点相连接，以实现网络的输入/输出:
x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")

# 为测试保存的模型，我们从MNIST数据库中取一个图像
image_b = mnist.test.images[100]
image_b=image_b.reshape(-1,28,28,1)
# 然后在选定的输入上运行保存的模型:
result = sess.run(y_conv, feed_dict={x:image_b})
print('Neural Network predicted', result[0])
print('Real label is:', np.argmax(mnist.test.labels[100]))
plt.imshow(image_b.reshape([28, 28]), cmap='Greys')
plt.show()