import tensorflow as tf
import numpy as np

#import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
labels = 10
imgW = 28
imgH = 28
picels = 28*28

x = tf.placeholder(tf.float32,[None,784])
label = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.truncated_normal([picels,labels]))
b = tf.Variable(tf.truncated_normal([labels]))
y = tf.matmul(x,W)+b

valid_prediction = tf.nn.softmax(
  tf.matmul(mnist.test.images, W) + b)
test_prediction = tf.nn.softmax(tf.matmul(mnist.test.images, W) + b)

#correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels = label,logits=y
))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

def accuracy(predictions,labels):
  a = np.argmax(predictions,1)
  b = np.argmax(labels,1)
  e = a == b
  c = np.sum(e)+0.0
  d = c / predictions.shape[0]
  return (100.0*d)

num_step = 8001
batch_size = 100
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for step in range(num_step):
    batch_data = mnist.train.next_batch(batch_size)
    _,l,predictions = sess.run([train_step,loss,y],feed_dict = {
      x:batch_data[0],
      label:batch_data[1]
    })
    if step %100 == 0:
      print('loss',l)
      print('Validation accuracy: %.lf%%' % accuracy(valid_prediction.eval(),mnist.test.labels))
  print('test accuracy: %.lf%%' % accuracy(test_prediction.eval(),mnist.test.labels))
