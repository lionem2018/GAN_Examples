import os, sys
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(dtype=np.float32).reshape(-1, 28, 28, 1)
y_train = y_train.astype(dtype=np.int32)
X_test = X_test.astype(dtype=np.float32).reshape(-1, 28, 28, 1)
y_test = y_test.astype(dtype=np.int32)

# training data에서 10000개의 데이터를 뽑음
val_indices = np.random.choice(range(X_train.shape[0]), size=10000, replace=False)
X_val = X_train[val_indices]
y_val = y_train[val_indices]
X_train = np.delete(arr=X_train, obj=val_indices, axis = 0)
y_train = np.delete(arr=y_train, obj=val_indices, axis = 0)
print(X_val.shape, y_val.shape, X_train.shape, y_val.shape)

train = zip(X_train, y_train)
validation = zip(X_val, y_val)
test = zip(X_test, y_test)

split = dict(train=train, validation=validation, test = test)

options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)

for key in split.keys():
    dataset = split.get(key)
    writer = tf.python_io.TFRecordWriter(path='./MNIST-data/tfrecords/mnist_{}.tfrecords'.format(key), options=options)

    for data, label in dataset:
        image = data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())
    else:
        writer.close()
        print('{} was converted to tfrecords'.format(key))


def parse_single_example(record):
    features = {'label' : tf.FixedLenFeature((), tf.int64, 0),
                'image' : tf.FixedLenFeature((), tf.string, '')}
    parsed_features = tf.parse_single_example(serialized = record, features = features)
    image = tf.decode_raw(parsed_features.get('image'), out_type = tf.float32)
    image = tf.reshape(tensor = image, shape = [28,28,1])
    label = tf.cast(parsed_features.get('label'), dtype = tf.int32)
    return image, label


n_epoch = 3
batch_size = 2

train = tf.data.TFRecordDataset(filenames='./MNIST-data/tfrecords/mnist_train.tfrecords', compression_type='GZIP')
train = train.map(lambda record: parse_single_example(record))
train = train.shuffle(buffer_size=30)
train = train.batch(batch_size)

val = tf.data.TFRecordDataset(filenames='./MNIST-data/tfrecords/mnist_validation.tfrecords', compression_type='GZIP')
val = val.map(lambda record: parse_single_example(record))
val = val.shuffle(buffer_size=30)
val = val.batch(10)

tr_iterator = train.make_initializable_iterator()
val_iterator = val.make_initializable_iterator()

X, y = val_iterator.get_next()

with tf.Session() as sess:
    sess.run(val_iterator.initializer)
    X_data = sess.run(X)
    y_data = sess.run(y)

print(X_data.shape, X_data.dtype)
print(y_data.shape, y_data.dtype)