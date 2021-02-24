import argparse
import os
import sys
import time
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import mnist
import input_data

FLAGS = None
#做完这些基本的操作之后我们就可以对其进行迭代训练了，但是在迭代训练之前我们得处理一下我们的输入数据，
# 之前我们一直都是用tf.placeholder来假装我们有输入数据，现在我们需要将真实数据传入进来：
#在开始训练之前我们需要先用tf.placeholder占位符帮手写体数据集占一个位置,
# 这里我们的标签的维度是[batch_size, 1]：
def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder

#fill_feed_dict函数输入是数据，图像placeholder的变量名称，和标签placeholder的变量名称，
# 返回一个字典，即需要传入到会话图中的数据。再接着就可以对其进行迭代训练：
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
    feed_dict = {images_pl: images_feed,labels_pl: labels_feed}
    return feed_dict

#评估模型的函数
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):

    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))

def run_training():
    #在开始训练之前，我们需要去读取数据：
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    # 在run_training()这个函数的一开始，是一个Python语言中的with命令，
    # 这个命令表明所有已经构建的操作都要与默认的tf.Graph全局实例关联起来。
    #tf.Graph实例是一系列可以作为整体执行的操作。TensorFlow的大部分场景只需要依赖默认图表一个实例即可。
    # 利用多个图表的更加复杂的使用场景也是可能的。
    with tf.Graph().as_default():
        # 我们需要调用之前写好的神经网络来对其进行训练，
        # 在训练之前需要调用之前写好的占位符函数placeholder_input（）来给输入数据占个位置：
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        #尽可能地构建好图表，满足促使神经网络向前反馈并做出预测的要求。
        #这样的话我们就假装我们有了输入的图片和其对应的标签
        #接下来我们构建图的输出公式：
        logits = mnist.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        #往inference图表中添加生成损失（loss）所需要的操作（ops）。
        #选择损失和训练参数
        loss = mnist.loss(logits, labels_placeholder)
        #往损失图表中添加计算并应用梯度（gradients）所需的操作。
        train_op = mnist.training(loss, FLAGS.learning_rate)
        #评估模型结果指标：
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        #把图运行过程中发生的事情(产生的数据记录下来)：
        summary = tf.summary.merge_all()
        #初始化变量：
        init = tf.global_variables_initializer()
        #建立一个保存训练中间数据的存档点：
        saver = tf.train.Saver()
        #建立会话：
        sess = tf.Session()
        #创建一个记事本写入器(这里我也想吐槽一下，为什么不之前的创建记事本放在一起)：
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        #之后再初始化变量：
        sess.run(init)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            # fill_feed_dict函数输入是数据，图像placeholder的变量名称，和标签placeholder的变量名称，返回一个字典，
            # 即需要传入到会话图中的数据。再接着就可以对其进行迭代训练：
            feed_dict = fill_feed_dict(data_sets.train,images_placeholder,labels_placeholder)
            # sess.run()会返回一个有两个元素的元组。其中每一个Tensor对象，对应了返回的元组中的numpy数组，
            # 而这些数组中包含了当前这步训练中对应Tensor的值。由于train_op并不会产生输出，
            # 其在返回的元祖中的对应元素就是None，所以会被抛弃。但是，如果模型在训练中出现偏差，
            # loss Tensor的值可能会变成NaN，所以我们要获取它的值，并记录下来。
            # 我们也希望记录一下程序运行的时间：start_time
            _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)
            duration = time.time() - start_time
            #如果进行顺利的话，我们每隔100步，打印一下步数，损失函数值，和程序运行的时间。
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            # 在每次运行summary时，都会往事件文件中写入最新的即时数据，
            # 函数的输出会传入事件文件读写器（writer）的add_summary()函数。
            # 我们也希能够保存我们的训练模型。
            # 每隔一千个训练步骤，我们的代码会尝试使用训练数据集与测试数据集，对模型进行评估。
            # do_eval函数会被调用三次，分别使用训练数据集、验证数据集与测试数据集。
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                #训练数据集：
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                #验证数据集
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                #测试数据集
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)





def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )

    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', 'MNIST_data')
                             ),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', './tmp')),
        help='Directory to put the log data.'
    )

    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'

    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)