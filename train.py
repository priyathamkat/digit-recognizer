from model import model

import numpy as np
import tensorflow as tf

import os
import sys
import time

TRAIN = 'train.csv'

tf.app.flags.DEFINE_string('data_directory', './data/', 'directory containing the data sets')
tf.app.flags.DEFINE_float('validation_fraction', 0.2, 'fraction of training data set aside for validation')
tf.app.flags.DEFINE_integer('batch_size', 1000, 'batch size for training')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'number of epochs trained')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'learning rate in training')
tf.app.flags.DEFINE_string('model_name', 'mnist', 'name of the saved model')
tf.app.flags.DEFINE_string('models_directory', './models/', 'directory to save the tensorflow model')

FLAGS = tf.app.flags.FLAGS


def train(train_images, train_labels):
    assert train_images.shape[1] == train_images.shape[2]
    train_size, image_size, _ = train_images.shape

    images_ph = tf.placeholder(tf.float32, shape=(None, image_size, image_size))
    labels_ph = tf.placeholder(tf.int64, shape=(None,))
    keep_prob = tf.placeholder(tf.float32)

    logits = model(images_ph, keep_prob, reuse=None)
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph))
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(logits, 1), labels_ph)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    train_op = optimizer.minimize(loss)
    init_op = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init_op)
        t0 = time.time()
        for e in range(FLAGS.num_epochs):
            for i in range(0, train_size, FLAGS.batch_size):
                batch_images = train_images[i:i + FLAGS.batch_size, :, :]
                batch_labels = train_labels[i:i + FLAGS.batch_size]

                step_loss, step_accuracy, _ = session.run([loss, accuracy, train_op], feed_dict={
                    images_ph: batch_images,
                    labels_ph: batch_labels,
                    keep_prob: 0.5
                })

                progress = ((e + (min(i + FLAGS.batch_size, train_size) / train_size)) / FLAGS.num_epochs)
                eta = (time.time() - t0) * (1 / progress - 1)

                print('progress: %.2f%%, loss: %.2f, accuracy: %.2f, eta: %.3f' %
                      (progress * 100, step_loss, 100 * step_accuracy, eta),
                      flush=True)

        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        save_path = os.path.join(FLAGS.models_directory, FLAGS.model_name + '.ckpt')
        saver.save(sess=session, save_path=save_path)
        print('saved tensorflow model at %s' % save_path, flush=True)


def validate(validation_images, validation_labels):
    assert validation_images.shape[1] == validation_images.shape[2]
    validation_size, image_size, _ = validation_images.shape

    images_ph = tf.placeholder(tf.float32, shape=(None, image_size, image_size))
    labels_ph = tf.placeholder(tf.int64, shape=(None,))
    keep_prob = tf.placeholder(tf.float32)

    logits = model(images_ph, keep_prob, reuse=True)
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(logits, 1), labels_ph)

    with tf.Session() as session:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        save_path = os.path.join(FLAGS.models_directory, FLAGS.model_name + '.ckpt')
        saver.restore(sess=session, save_path=save_path)

        weighted_sum = 0.0
        num_samples = 0
        t0 = time.time()
        for i in range(0, validation_size, FLAGS.batch_size):
            batch_images = validation_images[i:i + FLAGS.batch_size, :, :]
            batch_labels = validation_labels[i:i + FLAGS.batch_size]

            step_accuracy = session.run(accuracy, feed_dict={
                images_ph: batch_images,
                labels_ph: batch_labels,
                keep_prob: 1.0
            })

            weighted_sum += batch_images.shape[0] * step_accuracy
            num_samples += batch_images.shape[0]

            progress = min(i + FLAGS.batch_size, validation_size) / validation_size
            eta = (time.time() - t0) * (1 / progress - 1)

            sys.stdout.write('\r')
            sys.stdout.write('progress: %.2f%%, accuracy: %.2f, eta: %.2f' %
                             (progress * 100, 100 * step_accuracy, eta))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(_):
    train_path = os.path.join(FLAGS.data_directory, TRAIN)
    print('reading data from %s ...' % train_path, end='', flush=True)
    train_data = np.genfromtxt(train_path, dtype=np.uint8, delimiter=',', skip_header=1)
    print(' done', flush=True)

    labels = train_data[:, 0].astype(np.int64)

    image_size = int(np.sqrt(train_data.shape[1] - 1))
    images = train_data[:, 1:].reshape((-1, image_size, image_size)).astype(np.float32)
    images = np.array(images * 2.0 / 255 - 1, copy=False)

    train_size = int(np.round(images.shape[0] * (1 - FLAGS.validation_fraction)))
    train_images = images[:train_size, :, :]
    train_labels = labels[:train_size]
    print('beginning training...', flush=True)
    train(train_images, train_labels)
    print('training done', flush=True)

    validation_images = images[train_size:, :, :] if train_size != images.shape[0] else None
    validation_labels = labels[train_size:] if train_size != images.shape[0] else None
    if validation_images is not None:
        print('beginning validation...', flush=True)
        validate(validation_images, validation_labels)
        print('validation done', flush=True)

if __name__ == '__main__':
    tf.app.run()
