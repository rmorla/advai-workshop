import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.reset_default_graph()

# variables
x = tf.placeholder(tf.float32, [ None, 784])
y = tf.placeholder(tf.float32, [ None, 10])
w_1 = tf.Variable(tf.truncated_normal([784, 128]))
b_1 = tf.Variable(tf.truncated_normal([1, 128]))
w_2 = tf.Variable(tf.truncated_normal([128, 10]))
b_2 = tf.Variable(tf.truncated_normal([1, 10]))
# computational graph
def sigma(x):
   return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0),
        tf.exp(tf.negative(x))))
a_0 = x
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)
# a: output of the feedforward neural network with input data x
a = a_2
# accuracy
acct_mat = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))


# load images
mnist = input_data.read_data_sets("/tf/data/mnist/", one_hot=True)
images = mnist.test.images[:1000]
classes = mnist.test.labels[:1000]

# run session
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/tf/data/mnist-2-layer-model/model.ckpt")
    [acct_res_output, a_output] = sess.run([acct_res, a], feed_dict={ x : images, y : classes })
    print(acct_res_output)



