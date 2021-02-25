import tensorflow as tf
import numpy as np

def hello():
    hello = tf.constant("Hello TensorFlow!")
    sess=tf.Session()
    print(sess.run(hello))

    sess.close()
hello()

def x_X_y():
    x = tf.placeholder(tf.float32,[1],name="x")
    y = tf.placeholder(tf.float32,[1],name="y")
    z = tf.constant(1.0)
    y=x*z
    x_in = [100]
    sess=tf.Session()
    y_output = sess.run(y,{x:x_in})
    print(y_output)
    sess.close()
x_X_y()

def tensor_2d():
    tensor_2d = np.array([(1,2,3),(4,5,6),(7,8,9)])
    tensor_2d = tf.Variable(tensor_2d)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(tensor_2d.get_shape())
        print(sess.run(tensor_2d))
# (3, 3)
# [[1 2 3]
#  [4 5 6]
# [7 8 9]]
tensor_2d()

def fetch():
    constant_A = tf.constant([100.0])
    constant_B = tf.constant([300.0])
    constant_C = tf.constant([3.0])
    sum_ = tf.add(constant_A,constant_B)
    mul_ = tf.multiply(constant_A,constant_C)
    with tf.Session() as sess:
        result = sess.run([sum_,mul_])
        print(result)
        print(mul_)
        print(sum_)
# [array([400.], dtype=float32), array([300.], dtype=float32)]
# Tensor("Mul_1:0", shape=(1,), dtype=float32)
# Tensor("Add:0", shape=(1,), dtype=float32)
fetch()

def feed():
    a=3
    b=2
    x = tf.placeholder(tf.float32,shape=(a,b))
    y = tf.add(x,x)
    data = np.random.rand(a,b)
    sess = tf.Session()
    print(sess.run(y,feed_dict={x:data}))
# Tensor("Add:0", shape=(1,), dtype=float32)
# [[0.20300545 0.89083236]
#  [0.07491963 0.19097972]
# [1.8942114  0.48233417]]
feed()

def implementSignleNN():
    # 单输入神经元构建开始
    input_value = tf.constant(0.5,name="input_value")
    weight = tf.Variable(1.0,name="weight")
    expected_output = tf.constant(0.0,name="expected_output")
    model = tf.multiply(input_value, weight,"model")
    loss_function = tf.pow(expected_output - model,2,name="loss_function")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.025)
    # 单输入神经元构建结束
    for value in [input_value,weight,expected_output,model,loss_function]:
        tf.summary.scalar(value.op.name,value)

    summaries = tf.summary.merge_all()
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(100):
        summary_writer.add_summary(sess.run(summaries), i)
        sess.run(optimizer)
    # 完整神经元构建结束

    summaries = tf.summary.merge_all()
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)

    sess.run(tf.global_variables_initializer())
implementSignleNN()
