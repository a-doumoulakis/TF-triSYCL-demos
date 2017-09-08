from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

def main(_):

  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  conf = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.InteractiveSession(config=conf)
  with tf.device('/cpu:0'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    def weight_variable(shape):
      """Create a weight variable with appropriate initialization."""
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      """Create a bias variable with appropriate initialization."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
      """Reusable code for making a simple neural net layer.
      It does a matrix multiply, bias add, and then uses relu to nonlinearize.
      It also sets up name scoping so that the resultant graph is easy to read,
      and adds a number of summary ops. """

      with tf.device('/device:SYCL:0'):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = tf.matmul(input_tensor, weights) + biases
        activations = act(preactivate, name='activation')
        return activations

    hidden1 = nn_layer(x, 784, 484)

    dropped = tf.nn.dropout(hidden1, 0.9)

    y = nn_layer(dropped, 484, 10, act=tf.identity)

    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(diff)

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tf.global_variables_initializer().run()

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      xs, ys = mnist.train.next_batch(100)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys}

  for i in range(1000):
    if i % 10 == 0:
      acc = sess.run(accuracy, feed_dict=feed_dict(False))
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      sess.run(train_step, feed_dict=feed_dict(True))

  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/anastasi/MNIST/data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
