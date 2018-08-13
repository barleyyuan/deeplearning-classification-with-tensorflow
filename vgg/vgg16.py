"""
VGG-16 model defined in tensorflow
"""

import tensorflow as tf


class Vgg16(object):
    def __init__(self, num_classes, keep_prob):
        self.num_classes = num_classes
        self.keep_prob = keep_prob

    def model(self, x):
        with tf.variable_scope('vgg16'):
            x = self.conv_block(x, 64, 2, 'block1')
            x = self.conv_block(x, 128, 2, 'block2')
            x = self.conv_block(x, 256, 3, 'block3')
            x = self.conv_block(x, 512, 3, 'block4')
            x = self.conv_block(x, 512, 3, 'block5')
            x = tf.layers.flatten(x, name='flatten')
            x = self.fc_block(x, 4096, 1-self.keep_prob, 'fc6')
            x = self.fc_block(x, 4096, 1-self.keep_prob, 'fc7')
            self.logits = tf.layers.dense(x, units=self.num_classes, name='fc8/logits')
            self.prob = tf.layers.dense(x, units=self.num_classes, activation='softmax', name='prob')
        # return logits, prob

    def conv_block(self, x, filters, n_conv_layers, name):
        with tf.variable_scope(name):
            for i in range(n_conv_layers):
                x = self.conv2d(x, filters, 'conv%d' % (i+1))
            x = self.max_pool_2x2(x, 'max_pool')
        return x

    def fc_block(self, x, units, dropout_rate, name):
        with tf.variable_scope(name):
            x = tf.layers.dense(x, units=units, activation='relu', name='relu')
            x = tf.layers.dropout(x, rate=dropout_rate, name='dropout')
        return x

    def conv2d(self, inputs, filters, name):
        return tf.layers.conv2d(inputs, filters, kernel_size=3, strides=1, padding='same', activation='relu', name=name)

    def max_pool_2x2(self, inputs, name):
        return tf.layers.max_pooling2d(inputs, pool_size=2, strides=2, padding='same', name=name)


