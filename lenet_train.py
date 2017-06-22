import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import numpy as np
import os
import time
start = time.clock()

import lenet_inference

# 学习参数
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
REGULARIZER = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
BATCH_SIZE = 100

# 模型保存的路径名和文件名
MODEL_SAVE_PATH = '/path/to/model/'
MODEL_NAME = 'lenet.ckpt'

# 训练模型过程
def train(mnist):
    train = mnist.train

    x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, lenet_inference.IMAGE_SIZE, lenet_inference.IMAGE_SIZE, lenet_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(BATCH_SIZE, lenet_inference.OUTPUT_NODE), name='y-input')

    y = lenet_inference.inference(x, True, REGULARIZER)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.arg_max(y_, dimension=1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('loss', cross_entropy_mean)
    loss = tf.add_n(tf.get_collection('loss'))

    global_step = tf.Variable(0, trainable=False)
    # 虽然用不到shadow，但是会有，restore过程就可以用了
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate). \
        minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_op)

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAIN_STEPS):
            xs, ys = train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,
                                     newshape=(BATCH_SIZE,lenet_inference.IMAGE_SIZE,lenet_inference.IMAGE_SIZE,lenet_inference.NUM_CHANNELS))
            # sess.run(train_op, feed_dict={x: xs, y_: ys})
            # loss_value = sess.run(loss, feed_dict={x: xs, y_: ys})
            # step = sess.run(global_step)
            # 和下述代码应该是一样的
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            end = time.clock()
            if i%1000 == 0:
                print('After %d steps during %.6f sec, loss on training batch is %g' % (i, end-start, loss_value))

        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv=None):
    print(2)
    train(lenet_inference.mnist)

if __name__ == '__main__':
    print(1)
    tf.app.run(main)