from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Import data
    tf.logging.set_verbosity(tf.logging.DEBUG)
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    conf = tf.ConfigProto(allow_soft_placement=False, device_count={'SYCL': 2})
    sess = tf.InteractiveSession(config=conf)

    with tf.name_scope('basic_softmax_regression'):
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, [None, 784])
                y_ = tf.placeholder(tf.float32, [None, 10])

            with tf.name_scope('image_summary'):
                image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
                tf.summary.image('input', image_shaped_input, 5)

            with tf.device('/device:SYCL:0'):
                with tf.name_scope('weights'):
                    W = tf.Variable(tf.zeros([784, 10]))

            variable_summaries(W)
            with tf.device('/device:SYCL:0'):
                with tf.name_scope('biases'):
                    b = tf.Variable(tf.zeros([10]))

            variable_summaries(b)
            with tf.device('/device:SYCL:0'):
                with tf.name_scope('Wx'):
                    tmp = tf.matmul(x, W)

                with tf.name_scope('plus_b'):
                    y = tmp + b

            variable_summaries(y)

            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
                cross_entropy = tf.reduce_mean(diff)

            tf.summary.scalar('cross_entropy', cross_entropy)

            with tf.name_scope('train'):
                train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    tf.global_variables_initializer().run()

    for i in range(1000):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], {x: mnist.test.images, y_: mnist.test.labels})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={x: batch_xs, y_:batch_ys},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/anastasi/MNIST/data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='/home/anastasi/MNIST/logs/mnist_softmax',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
