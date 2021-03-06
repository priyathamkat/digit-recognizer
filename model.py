import tensorflow as tf


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def gated_pool_2x2(x, w):
    num_channels = tf.shape(x)[3]

    mask = tf.zeros([2, 2, num_channels, num_channels], dtype=tf.float32)
    mask = tf.matrix_set_diag(mask, tf.tile(tf.expand_dims(w, 2), [1, 1, num_channels]))

    a = tf.sigmoid(tf.nn.conv2d(x, mask, strides=[1, 2, 2, 1], padding='SAME'))
    return a * max_pool_2x2(x) + (1 - a) * avg_pool_2x2(x)


def model(x, keep_prob, reuse=None):
    with tf.variable_scope('mnist', reuse=reuse):
        image_size = tf.shape(x)[1]
        x = tf.reshape(x, [-1, image_size, image_size, 1])
        w_conv1 = tf.get_variable('w_conv1', [5, 5, 1, 32],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        b_conv1 = tf.get_variable('b_conv1', [32], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)

        w_pool1 = tf.get_variable('w_pool1', [2, 2],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        h_pool1 = tf.nn.dropout(gated_pool_2x2(h_conv1, w_pool1), keep_prob)

        w_conv2 = tf.get_variable('w_conv2', [5, 5, 32, 64], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(0, 0.1))
        b_conv2 = tf.get_variable('b_conv2', [64], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

        w_pool2 = tf.get_variable('w_pool2', [2, 2],
                                  dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        h_pool2 = tf.nn.dropout(gated_pool_2x2(h_conv2, w_pool2), keep_prob)

        w_fc1 = tf.get_variable('w_fc1', [7 * 7 * 64, 1024],
                                dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        b_fc1 = tf.get_variable('b_fc1', [1024], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = tf.get_variable('w_fc2', [1024, 10],
                                dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.1))
        b_fc2 = tf.get_variable('b_fc2', [10], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

        logits = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
        return logits
