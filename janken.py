"""
ref: http://qiita.com/sergeant-wizard/items/98ce0993a195475fd7a9, http://qiita.com/sergeant-wizard/items/af7a3bd90e786ec982d2
"""

import tensorflow as tf

input = [
  [1., 0., 0.],
  [0., 1., 0.],
  [0., 0., 1.]
]

winning_hands = [
  [0., 1., 0.],
  [0., 0., 1.],
  [1., 0., 0.]
]

def inference(input_placeholder):
  W = tf.Variable(tf.zeros([3, 3]))
  b = tf.Variable(tf.zeros([3]))

  y = tf.nn.softmax(tf.matmul(input_placeholder, W) + b)
  return y

def loss(output, supervisor_labels_placeholder):
  cross_entropy = -tf.reduce_sum(supervisor_labels_placeholder * tf.log(output))
  tf.summary.scalar("entropy", cross_entropy)
  return cross_entropy

def training(loss):
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  return train_step

supervisor_labels_placeholder = tf.placeholder("float", [None, 3])
input_placeholder = tf.placeholder("float", [None, 3])
feed_dict={input_placeholder: input, supervisor_labels_placeholder: winning_hands}
summary_writer = tf.summary.FileWriter('data')

with tf.Session() as sess:
  output = inference(input_placeholder)
  loss = loss(output, supervisor_labels_placeholder)
  training_op = training(loss)

  summary_op = tf.summary.merge_all()

  init = tf.global_variables_initializer()
  sess.run(init)

  for step in range(1000):
    sess.run(training_op, feed_dict=feed_dict)
    if (step % 100 == 0):
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
