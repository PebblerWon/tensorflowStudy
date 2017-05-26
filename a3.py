# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_actual = tf.placeholder(tf.float32,[None,10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  biases = tf.constant(0.1,shape=shape)
  return tf.Variable(biases)

def conv2d(x,w):
  return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#池化
def max_pool(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1, 2, 2, 1], padding='SAME')

#构建网络

#第一层卷积层
x_image = tf.reshape(x,[-1,28,28,1])
w1 =tf.Variable( tf.truncated_normal([5,5,1,32],stddev=0.1))
b1 =tf.Variable( tf.constant(0.1,shape=[32]))
conv2d1 = tf.nn.conv2d(x_image,w1,strides=[1,1,1,1],padding='SAME')+b1
hconv1 = tf.nn.relu(conv2d1)
hpool1 = tf.nn.max_pool(hconv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第二层卷积层
w2 =tf.Variable( tf.truncated_normal([5,5,32,64],stddev=0.1))
b2 =tf.Variable( tf.constant(0.1,shape=[64]))
conv2d2 = tf.nn.conv2d(hpool1,w2,strides=[1,1,1,1],padding='SAME')+b2
hconv2 = tf.nn.relu(conv2d2)
hpool2 = tf.nn.max_pool(hconv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第三层
kernel3 =tf.Variable( tf.truncated_normal([7*7*64,1024],stddev=0.1))
b3 =tf.Variable( tf.constant(0.1,shape=[1024]))
h_pool2_flat = tf.reshape(hpool2,[-1,7*7*64])
conv2d3 = tf.matmul(tf.reshape(hpool2,[-1,7*7*64]),kernel3)+b3
h3 = tf.nn.relu(conv2d3)

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h3, keep_prob)                  #dropout层

#连接层
#不加方差训练结果很差
#w_f =tf.Variable( tf.truncated_normal([1024,10]))
w_f =tf.Variable( tf.truncated_normal([1024,10],stddev=0.1))
b_f =tf.Variable( tf.constant(0.1,tf.float32,shape=[10]))
y_f = tf.matmul(h_fc1_drop,w_f)+b_f
y_predict = tf.nn.softmax(y_f)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels = y_actual,logits=y_f
))
cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))

#train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
#train_step=tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))             

training_times = 2000
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for i in range(training_times):
    train_batch = mnist.train.next_batch(50)
    if(i%100 ==0):
      test_batch = mnist.test.next_batch(100)
      train_acc = accuracy.eval(feed_dict={x:test_batch[0], y_actual: test_batch[1], keep_prob: 1.0})
      print('step %d : training accuracy %.1f%%' % (i,train_acc*100.0))
    train_step.run(feed_dict={x:train_batch[0],y_actual:train_batch[1], keep_prob: 0.5})

  test_acc=accuracy.eval(feed_dict={x:mnist.test.images[:2000], y_actual: mnist.test.labels[:2000], keep_prob: 1.0})
  print("test accuracy %.1f%%" % (test_acc*100.0))
  
  saver = tf.train.Saver()
  save_path = saver.save(sess,"a3Ckpt/convValue.ckpt")
  print("save to path",save_path)
  try:
    f = open('a3.pickle', 'wb')
    save = {
      'w1': sess.run(w1),
      'b1': sess.run(b1)
      }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print("over")
