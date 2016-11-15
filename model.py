import tensorflow as tf


def model(x, reuse=None):
    with tf.variable_scope('mnist', reuse=reuse):
        size = tf.shape(x)[0]
        x = tf.reshape(x, [size, 784])
        w = tf.get_variable('w', [784, 10], dtype=tf.float32)
        b = tf.get_variable('b', [10], dtype=tf.float32)
        logits = tf.matmul(x, w) + b
        return logits
