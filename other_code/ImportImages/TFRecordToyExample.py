import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

print(mnist.train.images.shape, mnist.train.images.dtype)
print(mnist.train.labels.shape, mnist.train.labels.dtype)

# tfrecord encode, decode를 위해 data type과 length 확인
train_data_length = mnist.train.images.shape[0]
test_data_length = mnist.test.images.shape[0]
print('train data length', train_data_length)
print('test data length', test_data_length)
# nums= mnist.train.images[0:5]
# plt.figure(figsize = (5,1))
# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.axis('off')
#     plt.imshow(np.reshape(nums[i],(28,28)),cmap='gray')

# train 데이터 tfrecord 파일로 생성
writer = tf.python_io.TFRecordWriter('./MNIST-data/train.tfrecord')
for idx in range(train_data_length):
    tfrecord_obj = tf.train.Example(features=tf.train.Features(feature={
 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.train.images[idx].tostring()])),
 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.train.labels[idx].tostring()])),
 }))
    writer.write(tfrecord_obj.SerializeToString())
writer.close()

writer = tf.python_io.TFRecordWriter('./MNIST-data/test.tfrecord')
for idx in range(test_data_length):
    tfrecord_obj = tf.train.Example(features=tf.train.Features(feature={
 'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.test.images[idx].tostring()])),
 'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.test.labels[idx].tostring()])),
 }))
    writer.write(tfrecord_obj.SerializeToString())
writer.close()


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


train_dataset = tf.data.TFRecordDataset("./MNIST-data/train.tfrecord")
train_dataset = train_dataset.map(decode)
train_dataset = train_dataset.repeat()

test_dataset = tf.data.TFRecordDataset("./MNIST-data/test.tfrecord")
test_dataset = test_dataset.map(decode)
test_dataset = test_dataset.repeat()

iterator = train_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
for i in range(train_data_length):
    image, label = sess.run(next_element)
    assert(np.array_equal(image, mnist.train.images[i]))
sess.close()
