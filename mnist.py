#coding: utf-8

# MNIST データセットを読み込む
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
 
# 入力データ格納用の 784 px 分のプレースホルダを作成
x = tf.placeholder(tf.float32, [None, 784])
 
# 重み (784 x 10 の行列) の Variable を定義
W = tf.Variable(tf.zeros([784, 10]))
 
# バイアス (長さ 10 の行列) の Variable を定義
b = tf.Variable(tf.zeros([10]))
 
# ソフトマックス回帰による予測式を定義
y = tf.nn.softmax(tf.matmul(x, W) + b)
 
# 出力データ (予測値) 格納用のプレースホルダ
y_ = tf.placeholder(tf.float32, [None, 10])
 
# 交差エントロピーを最小化するよう、学習を行う式を以下のように定義
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
 
# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 学習を 1,000 回繰り返す
for i in range(1000):
  # 訓練用データから 100 件取得
  batch_xs, batch_ys = mnist.train.next_batch(100)
  # train_step を実行
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 実際の値と予測された値が同じであるか確認
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
# 平均値を求め、予測精度を求める
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# x にテストデータ、y_ に実際の値をあてはめ、上記で作成した式を実行
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
