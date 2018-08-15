import tensorflow as tf


class Resnet(object):
    def __init__(self, depth, num_classes):
        self.depth = depth
        self.num_classes = num_classes

    def model(self, x):
        with tf.variable_scope('ResNet'):
            net = tf.layers.conv2d(inputs=x,
                                   filters=64,
                                   kernel_size=7,
                                   strides=2,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='conv1',
                                   )
            net = tf.layers.max_pooling2d(inputs=net,
                                          pool_size=3,
                                          strides=2,
                                          padding='same',
                                          name='maxpool1')
            if self.depth == 18:
                net = self.block(net, 64, 2, 'block2', 'shallow', strides=1)
                net = self.block(net, 128, 2, 'block3', 'shallow')
                net = self.block(net, 256, 2, 'block4', 'shallow')
                net = self.block(net, 512, 2, 'block5', 'shallow')
            elif self.depth == 34:
                net = self.block(net, 64, 3, 'block2', 'shallow', strides=1)
                net = self.block(net, 128, 4, 'block3', 'shallow')
                net = self.block(net, 256, 6, 'block4', 'shallow')
                net = self.block(net, 512, 3, 'block5', 'shallow')
            elif self.depth == 50:
                net = self.block(net, 256, 3, 'block2', 'deep', strides=1)
                net = self.block(net, 512, 4, 'block3', 'deep')
                net = self.block(net, 1024, 6, 'block4', 'deep')
                net = self.block(net, 2048, 3, 'block5', 'deep')
            elif self.depth == 101:
                net = self.block(net, 256, 3, 'block2', 'deep', strides=1)
                net = self.block(net, 512, 4, 'block3', 'deep')
                net = self.block(net, 1024, 23, 'block4', 'deep')
                net = self.block(net, 2048, 3, 'block5', 'deep')
            elif self.depth == 152:
                net = self.block(net, 256, 3, 'block2', 'deep', strides=1)
                net = self.block(net, 512, 8, 'block3', 'deep')
                net = self.block(net, 1024, 36, 'block4', 'deep')
                net = self.block(net, 2048, 3, 'block5', 'deep')
            else:
                raise ValueError("Expected argument 'deep' must be in [18, 34, 50, 101, 152].")

            net = tf.layers.average_pooling2d(inputs=net,
                                              pool_size=7,
                                              strides=1,
                                              name='avgpool1'
                                              )
            net = tf.layers.flatten(inputs=net)
            self.logits = tf.layers.dense(inputs=net,
                                     units=self.num_classes,
                                     name='logits',
                                     )
            self.prob = tf.nn.softmax(self.logits, name='prob')

    def block(self, x, n_out, n_bottleneck, scope, mode, strides=2):
        with tf.variable_scope(scope):
            out = self.bottleneck(x, n_out=n_out, scope="bottleneck1", mode=mode, strides=strides)
            for i in range(1, n_bottleneck):
                out = self.bottleneck(out, n_out, scope="bottleneck{0}".format(i + 1), mode=mode)
        return out

    def bottleneck(self, x, n_out, scope, mode, strides=1):
        n_in = x.get_shape()[-1]
        n_first = n_out / 4
        with tf.variable_scope(scope):
            if mode == 'shallow':
                f = tf.layers.conv2d(inputs=x,
                                     filters=n_out,
                                     kernel_size=3,
                                     strides=strides,
                                     padding='same',
                                     activation=tf.nn.relu,
                                     name='conv1'
                                     )
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_out,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     activation=None,
                                     name='conv2'
                                     )
            elif mode == 'deep':
                f = tf.layers.conv2d(inputs=x,
                                     filters=n_first,
                                     kernel_size=1,
                                     strides=strides,
                                     name='conv1',
                                     )
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_first,
                                     kernel_size=3,
                                     strides=1,
                                     padding='same',
                                     name='conv2',
                                     )
                f = tf.layers.conv2d(inputs=f,
                                     filters=n_out,
                                     kernel_size=1,
                                     strides=1,
                                     name='conv3')
            else:
                raise ValueError("expected argument 'mode' must be between 'shallow' and 'deep'.")

            if n_in != n_out:  # projection
                shortcut = tf.layers.conv2d(inputs=x,
                                            filters=n_out,
                                            kernel_size=1,
                                            strides=strides,
                                            name='projection')
            else:
                shortcut = x  # identical mapping
            out = tf.nn.relu(shortcut + f)
        return out
