# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  with tf.device('/cpu:0'):
    # All operations will run on the cpu  device unless stated otherwise

    # Describe the model
    # x will hold the vectorized image that we want to recognize
    x = tf.placeholder(tf.float32, [None, 784])
    # y_ will hold the correct answer, it will hold nine 0's and one 1. If y_ = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] then the input image was a 3 
    y_ = tf.placeholder(tf.float32, [None, 10])

    # W and b are the weights and biases that will be infered by tensorflow
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    with tf.device('/device:SYCL:0'):
      # This operation will run on the first SYCL device
      #
      # We multiply our image vector by the weights and add the bias
      #
      # After this operation y will hold 10 values, each value is the 'evidence' that the input image is one of the output classes
      # Softmax will then turn these 10 values into probabilities that will add up to one giving us a prediction of the digit described in x
      y = tf.matmul(x, W) + b

    # We now need to define a correctness notion for our model (the loss function), we use the cross-entropy
    # The cross-entropy is a measurement of how inefficient our predictions are for describing the truth
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # Then we use the backpropagation algorithm to efficiently determine how our weights and biases affect the cross_entropy
    # We optimize the backpropagation with the gradient descent algorithm to help minimize the cross_entropy
    #
    # With this tensorflow adds new operations to our graph which implement backpropagation and gradient descent
    # Then it gives us back a single operation which, when run, does a step of gradient descent training
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # We compute the accuracy of our model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  # We configure and create a session that will run the model
  # allow_soft_placement allows tensorflow to use a fallback device if an operation is not available for the stated device
  conf = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.InteractiveSession(config=conf)
  tf.global_variables_initializer().run()

  # Train
  for i in range(1000):

    if i % 10 == 0:
      # Every 10 iteration we check the accuracy
      print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

    # We get a batch of 100 images whith which we will train the model
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # We fetch the next training step for our training
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test the trained model, we fetch the accuracy and print it
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/anastasi/MNIST/data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
