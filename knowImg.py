# -*- coding:utf-8 -*-
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import os
import sys
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image

#imgPath = input('please enter your file path: ')
#print('hello,',imgPath)
pixel_depth = 255
imgW = 28
imgH = 28
#image_data = (ndimage.imread('imgsWhiteBg/0.png').astype(float) - 
#                    pixel_depth / 2) / pixel_depth

#image_data = ndimage.imread('imgsBlackBg/0.png').astype(float)
img = Image.open('imgsWhiteBg/9.png').convert('L')

# resize的过程
if img.size[0] != 28 or img.size[1] != 28:
    img = img.resize((28, 28))

# 暂存像素值的一维数组
arr = []

for i in range(28):
    for j in range(28):
        # mnist 里的颜色是0代表白色（背景），1.0代表黑色
        pixel = 1.0 - float(img.getpixel((j, i)))/255.0
        # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
        arr.append(pixel)

arr1 = np.array(arr).reshape((1, 28, 28, 1)).astype(np.float32)

x_image = tf.reshape(arr1,[-1,28,28,1])
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

saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess,"a3Ckpt/convValue.ckpt")
  predict =tf.argmax(tf.nn.softmax(tf.matmul(h_fc1_drop,w_f)+b_f),1)
  #predict =tf.nn.softmax(tf.matmul(h_fc1_drop,w_f)+b_f)
  value = predict.eval(feed_dict={keep_prob: 0.5})
  print(value)
#print(image_data)
