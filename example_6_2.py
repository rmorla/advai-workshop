import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt

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


# define gradients
def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))
delta_2 = tf.multiply(tf.subtract(a_2, y), sigmaprime(z_2))
d_b_2 = delta_2
d_w_2 = tf.matmul(tf.transpose(a_1), delta_2)
delta_1 = tf.matmul(delta_2, tf.transpose(w_2))
delta_1 = tf.multiply(sigmaprime(z_1), delta_1)
d_b_1 = delta_1
d_w_1 = tf.matmul(tf.transpose(a_0), delta_1)

#create noisy input data
x_noise = tf.truncated_normal([1, 784])
eta = tf.constant(0.01)
x_star = tf.clip_by_value(tf.add(x, tf.multiply(eta, tf.sign(x_noise))), 0.0, 1.0)         

# load images
mnist = input_data.read_data_sets("/tf/data/mnist/", one_hot=True)
ind = 3
images = mnist.test.images[ind:ind+1]
classes = mnist.test.labels[ind:ind+1]


# run session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "/tf/data/mnist-2-layer-model/model.ckpt")
    x_star_value = images
    y_star_value = sess.run(a, feed_dict={ x : x_star_value})
    print (np.argmax(y_star_value, axis=1), np.argmax(classes, axis = 1))
    for i in range(300):
        x_star_value = sess.run(x_star, feed_dict={ x : x_star_value, y : classes })
        y_star_value = sess.run(a, feed_dict={ x : x_star_value })
        if np.argmax(classes, axis = 1) != np.argmax(y_star_value, axis=1):
            print ('Classification of image changed after {i} iterations'.format(i=i))
            break

    plt.title('Original image label is {label}'.format(label=np.argmax(classes)))
    plt.imshow(images.reshape(28, 28), cmap='gray')
    plt.show()
    
    plt.title('Attack image label is {label}'.format(label=np.argmax(y_star_value)))
    plt.imshow(x_star_value.reshape(28, 28), cmap='gray')
    plt.show()    
