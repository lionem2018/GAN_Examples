import tensorflow as tf
import matplotlib.pyplot as plt


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
    })
    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.uint8)

    # image = tf.cast(image, tf.int32)
    # image = tf.reshape(image, shape)
    return image, label


filenames = tf.placeholder(tf.string, shape=[None])
training_filenames = ['./MNIST-data/tfrecords/train.tfrecord', './MNIST-data/tfrecords/test.tfrecord']

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(decode)
dataset = dataset.shuffle(buffer_size=30)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

next_element = iterator.get_next()

# You can feed the initializer with the appropriate filenames for the current
# phase of execution, e.g. training vs. validation.

# Initialize `iterator` with training data.
with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    X_data, y_data = sess.run(next_element)

    for i in range(32):

        if i % 8 == 0:
            plt.imshow(X_data[i].reshape(28, 28,), interpolation='nearest')
            plt.show()
