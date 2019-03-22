import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from ops import *
from tensorflow.layers import batch_normalization
from tensorflow.keras.layers import UpSampling2D

class Generator:
    def __init__(self, X, Y, img_shape, batch_size):
        self.img_rows, self.img_cols, self.channels = img_shape
        self.batch_size = batch_size
        with tf.variable_scope('g'):
            print("Initializing generator weights")
            self.W1 = init_weights([X+Y, 16*16*512])
            self.W2 = init_weights([3, 3, 512, 256])
            self.W3 = init_weights([3, 3, 256, 128])
            self.W4 = init_weights([3, 3, 128, 1])
            

    def forward(self, X, Y, momentum=0.5):

        # print('X shape in G is ', X.shape)
        # print('Y shape in G is ', Y.shape)
        # print('X type in G is ', type(X))
        # print('Y type in G is ', type(Y))
        # print('X dtype in G is ', X.dtype)
        # print('Y dtype in G is ', Y.dtype)
        # print('W1 shape in G is ', self.W1.shape)

        z = tf.concat([X, Y], 1)
        # print('z shape in G after concat is ', z.shape)
        # print('z type in G is ', type(z))
        # print('z dtype in G is ', z.dtype)

        z = tf.matmul(z, self.W1)
        z = tf.nn.relu(z)
        z = tf.reshape(z, [-1, 16, 16, 512])

        z = UpSampling2D()(z)
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME")
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = UpSampling2D()(z)
        z = conv2d(z, self.W3, [1, 1, 1, 1], padding="SAME") 
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME") 

        return tf.nn.tanh(z)

