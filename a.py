import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot = True)
labels = 10
imgW = 28
imgH = 28
picels = 28*28
train = mnist.train.next_batch(5000)
#x = tf.placeholder(tf.float32,[None,784])
#y_ = tf.placeholder(tf.float32,[None,10])
x= train[0]
y_ = train[1]

W = tf.Variable(tf.truncated_normal([picels,labels]))
b = tf.Variable(tf.truncated_normal([labels]))
y = tf.matmul(x,W)+b

test_validcation = tf.nn.softmax(tf.matmul(mnist.test.images,W)+b)
#correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels = y_,logits=y
))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
def accuracy(predictions,labels):
  a = np.argmax(predictions,1)
  b = np.argmax(labels,1)
  e = a == b
  c = np.sum(e)+0.0
  d = c / predictions.shape[0]
  return (100.0*d)
num_steps = 3000
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  
  for step in range(num_steps):
    _,l,predictions = sess.run([train_step,loss,y])
    if step %100 == 0:
      print('loss',l)
      print('Validation accuracy: %.lf%%' % accuracy(test_validcation.eval(),mnist.test.labels));
    #train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
