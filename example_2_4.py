#2.4

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("/tf/data/mnist/", one_hot=True)
image = mnist.test.images[0]
image_class = mnist.test.labels[0]
image_2d = image.reshape(28, 28)
plt.title('Label is {label}'.format(label=image_class ))
plt.imshow(image_2d, cmap='gray')
plt.show()


