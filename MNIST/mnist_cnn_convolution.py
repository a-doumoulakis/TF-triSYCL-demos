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
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  conf = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.InteractiveSession(config=conf)
  # Create a multilayer model.
  with tf.name_scope('advanced_cnn_softmax_regression'):
    with tf.device('/cpu:0'):
      # Input placeholders
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

      x_image = tf.reshape(x, [-1, 28, 28, 1])
 
      with tf.name_scope('image_summary'):
        tf.summary.image('input', x_image, 1)

      # We can't initialize these variables to 0 - the network will get stuck.
      def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

      def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

      def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

      def avg_pool_2x2(x):
        with tf.device('/device:SYCL:0'):
          return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

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


      def nn_layer1(input_tensor, weight_shape, bias_shape, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
          with tf.device('/device:SYCL:0'):
            with tf.name_scope('weights'):
              weights = weight_variable(weight_shape)
              variable_summaries(weights)
            with tf.name_scope('biases'):
              biases = bias_variable(bias_shape)
              variable_summaries(biases)
            with tf.name_scope('conv_and_max_pool'):
              h_conv = act(conv2d(input_tensor, weights) + biases)
              h_pool = max_pool_2x2(h_conv)
            tf.summary.histogram('after_first_layer_type', h_pool)
            return h_pool


      def nn_layer2(input_tensor, weight_shape, bias_shape, layer_name, act=tf.nn.relu):
        with tf.name_scope(layer_name):
          with tf.name_scope('weights'):
            weights = weight_variable(weight_shape)
            variable_summaries(weights)
          with tf.name_scope('biases'):
            biases = bias_variable(bias_shape)
            variable_summaries(biases)
          with tf.name_scope('conv_and_avg_pool'):
            h_pool = avg_pool_2x2(input_tensor)
            h_conv = act(conv2d(h_pool, weights) + biases)
          tf.summary.histogram('after_second_layer_type', h_pool)
          return h_conv

      hidden10 = nn_layer1(x_image, [5, 5, 1, 32], [32], 'layer-1-0')

      hidden20 = nn_layer1(x_image, [5, 5, 1, 16], [16], 'layer-2-0')

      hidden11 = nn_layer1(hidden10, [5, 5, 32, 64], [64], 'layer-1-1')

      hidden21 = nn_layer2(hidden20, [5, 5, 16, 64], [64], 'layer-2-1')

      with tf.name_scope('image_summary'):
        with tf.name_scope('image_layer-1-0'):
          x_img_lay11 = tf.reshape(tf.slice(hidden10, [0,0,0,0], [1,14,14,1]), [-1, 14, 14, 1])
          tf.summary.image('layer-1-0', x_img_lay11, 1)

        with tf.name_scope('image_layer-2-0'):
          x_img_lay21 = tf.reshape(tf.slice(hidden20, [0,0,0,0], [1,14,14,1]), [-1, 14, 14, 1])
          tf.summary.image('layer-2-0', x_img_lay21, 1)

        with tf.name_scope('image_layer-1-1'):
          x_img_lay11 = tf.reshape(tf.slice(hidden11, [0,0,0,0], [1,7,7,1]), [-1, 7, 7, 1])
          tf.summary.image('layer-1-1', x_img_lay11, 1)

        with tf.name_scope('image_layer-2-1'):
          x_img_lay21 = tf.reshape(tf.slice(hidden21, [0,0,0,0], [1,7,7,1]), [-1, 7, 7, 1])
          tf.summary.image('layer-2-1', x_img_lay21, 1)

      with tf.name_scope('concat'):
        h_pool11_flat = tf.reshape(hidden11, [-1, 7*7*64])
        h_pool21_flat = tf.reshape(hidden21, [-1, 7*7*64])
        hidden_concat = tf.concat([h_pool11_flat, h_pool21_flat], 1)
      tf.summary.histogram('concat', hidden_concat)

      with tf.name_scope('densely_connected'):
        W_fc1 = weight_variable([(7 * 7 * 64) + (7 * 7 * 64), 2048])
        variable_summaries(W_fc1)

        b_fc1 = bias_variable([2048])
        variable_summaries(b_fc1)

        h_fc1 = tf.nn.relu(tf.matmul(hidden_concat, W_fc1) + b_fc1)

      tf.summary.histogram('densely_connected', h_fc1)

      with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

      tf.summary.histogram('dropout', h_fc1_drop)

      with tf.name_scope('readout'):
        W_fc2 = weight_variable([2048, 10])
        variable_summaries(W_fc2)
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2)
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

      tf.summary.histogram('readout', y_conv)

      with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(diff)

      tf.summary.scalar('cross_entropy', cross_entropy)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

      with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      tf.summary.scalar('accuracy', accuracy)

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    train_writer.flush()
    test_writer.flush()
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/home/anastasi/MNIST/data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/home/anastasi/MNIST/logs/mnist_cnn_convolution',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
