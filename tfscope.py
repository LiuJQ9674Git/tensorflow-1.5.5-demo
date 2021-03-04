# 变量共享主要涉及两个函数：
# tf.get_variable(<name>, <shape>, <initializer>)
# tf.variable_scope(<scope_name>)
# 即就是必须要在tf.variable_scope的作用域下使用tf.get_variable()函数。
# 这里用tf.get_variable( ) 而不用tf.Variable( )，是因为前者拥有一个变量检查机制，
# 会检测已经存在的变量是否设置为共享变量，如果已经存在的变量没有设置为共享变量，
# TensorFlow 运行到第二个拥有相同名字的变量的时候，就会报错。

# 两个创建变量的方式。如果使用tf.Variable() 的话每次都会新建变量。
# 但是大多数时候我们是希望重用一些变量，所以就用到了get_variable()，
# 它会去搜索变量名，有就直接用，没有再新建。
# 名字域。既然用到变量名了，就涉及到了名字域的概念。

# 这就是为什么会有scope 的概念。
# name_scope 作用于操作，variable_scope 可以通过设置reuse标志以及初始化方式来影响域下的变量。

# 在tf.variable_scope的作用域下，通过get_variable()使用已经创建的变量，实现了变量的共享。
# 在 train RNN 和 test RNN 的时候, RNN 的 time_steps 会有不同的取值, 这将会影响到整个 RNN 的结构,
# 所以导致在 test 的时候, 不能单纯地使用 train 时建立的那个 RNN.
# 但是 train RNN 和 test RNN 又必须是有同样的 weights biases 的参数.
# 所以, 这时, 就是使用 reuse variable 的好时机.

class TrainConfig:
    batch_size = 20

time_steps = 20
input_size = 10
output_size = 2
cell_size = 11
learning_rate = 0.01

class TestConfig(TrainConfig):
    time_steps = 1

train_config = TrainConfig()
test_config = TestConfig()

# 并且定义 scope.reuse_variables(),
# 使我们能把 train_rnn 的所有 weights, biases参数全部绑定到 test_rnn 中

# 这样, 不管两者的 time_steps 有多不同, 结构有多不同, train_rnn W, b 参数更新成什么样,
# test_rnn 的参数也更新成什么样.


