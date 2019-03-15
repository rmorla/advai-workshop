import tensorflow as tf
# define computational graph
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)
# process data using previously defined computational graph
with tf.Session() as sess:
    add_result = sess.run(add, feed_dict={a: 2, b: 3})
    mul_result = sess.run(mul, feed_dict={a: 2, b: 3})
    print (add_result, mul_result)
