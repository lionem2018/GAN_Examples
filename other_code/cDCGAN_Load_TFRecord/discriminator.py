import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from ops import *
from tensorflow.layers import batch_normalization


class Discriminator:
    def __init__(self, X, Y, img_shape):

        self.img_rows, self.img_cols, self.channels = img_shape
        with tf.variable_scope('d'):
            print("Initializing discriminator weights")
            self.W1 = init_weights([5, 5, X+Y, 64])
            self.b1 = init_bias([64])
            self.W2 = init_weights([3, 3, 64, 64])
            self.b2 = init_bias([64])
            self.W3 = init_weights([3, 3, 64, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([2, 2, 128, 256])
            self.b4 = init_bias([256])
            self.W5 = init_weights([16*16*256, 1])
            self.b5 = init_bias([1])
            
    
    def forward(self, X, Y, momentum=0.5):

        # print('X shape in D is ', X.shape)
        # print('Y shape in D is', Y.shape)
        # print('X type in D is ', type(X))
        # print('Y type in D is ', type(Y))
        # print('X dtype in D is ', X.dtype)
        # print('Y dtype in D is', Y.dtype)
        # print('W1 shape is D ', self.W1.shape)

        X = tf.concat([X, Y], 3)
        # print('X shape after concat  in D', X.shape)
        # print('X type after concat in D is ', type(X))
        # print('X dtype after concat  in D', X.dtype)
        #shape of input = [batch, in_height, in_width, in_channels]
        #shape of filter = [filter_height, filter_width, in_channels, out_channels]

        #Last dimension of input and third dimension of filter represents the number of input channels. 
        #They shoudl be equal.
        z = conv2d(X, self.W1, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b1)
        z = tf.nn.leaky_relu(z)
        
        z = conv2d(z, self.W2, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b2)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W3, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b3)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)

        z = conv2d(z, self.W4, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.b4)
        z = batch_normalization(z, momentum=momentum)
        z = tf.nn.leaky_relu(z)
        
        z = tf.reshape(z, [-1, 16*16*256])
        logits = tf.matmul(z, self.W5)
        logits = tf.nn.bias_add(logits, self.b5)
        return logits  #shape of this logit is (?, 1)
