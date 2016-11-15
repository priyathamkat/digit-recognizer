from model import model

import numpy as np
import tensorflow as tf

import os
import sys
import time

TEST = 'test.csv'
LABELS = 'labels.csv'

tf.app.flags.DEFINE_string('data_directory', './data/', 'directory containing the data sets')
tf.app.flags.DEFINE_integer('batch_size', 1000, 'batch size for training')
tf.app.flags.DEFINE_string('model_name', 'mnist', 'name of the saved model')
tf.app.flags.DEFINE_string('models_directory', './models/', 'directory to save the tensorflow model')

FLAGS = tf.app.flags.FLAGS


def test(images):
    assert images.shape[1] == images.shape[2]
    images_size, image_size, _ = images.shape

    test_labels = np.zeros(images_size, dtype=np.int64)
    images_ph = tf.placeholder(tf.float32, shape=(None, image_size, image_size))
    keep_prob = tf.placeholder(tf.float32)

    logits = model(images_ph, keep_prob, reuse=None)
    labels = tf.argmax(logits, 1)

    with tf.Session() as session:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        save_path = os.path.join(FLAGS.models_directory, FLAGS.model_name + '.ckpt')
        saver.restore(sess=session, save_path=save_path)

        t0 = time.time()
        for i in range(0, images_size, FLAGS.batch_size):
            batch_images = images[i:i + FLAGS.batch_size, :, :]

            step_labels = session.run(labels, feed_dict={
                images_ph: batch_images,
                keep_prob: 1.0
            })
            test_labels[i:i + FLAGS.batch_size] = step_labels
            progress = min(i + FLAGS.batch_size, images_size) / images_size
            eta = (time.time() - t0) * (1 / progress - 1)

            sys.stdout.write('\r')
            sys.stdout.write('progress: %.2f%%, eta: %.2f' %
                             (progress * 100, eta))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return test_labels


def main(_):
    test_path = os.path.join(FLAGS.data_directory, TEST)
    print('reading data from %s ...' % test_path, end='', flush=True)
    train_data = np.genfromtxt(test_path, dtype=np.uint8, delimiter=',', skip_header=1)
    print(' done', flush=True)

    image_size = int(np.sqrt(train_data.shape[1]))
    images = train_data.reshape((-1, image_size, image_size)).astype(np.float32)
    images = np.array(images * 2.0 / 255 - 1, copy=False)

    labels = test(images)
    ids = np.arange(1, labels.shape[0] + 1, dtype=np.int64)
    save_path = os.path.join(FLAGS.data_directory, LABELS)
    np.savetxt(save_path,
               np.vstack((ids, labels)).transpose(),
               fmt='%d',
               delimiter=',',
               header='ImageId,Label',
               comments='')

if __name__ == '__main__':
    tf.app.run()
