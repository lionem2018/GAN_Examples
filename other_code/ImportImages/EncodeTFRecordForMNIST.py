import tensorflow as tf

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
writer = tf.python_io.TFRecordWriter('./MNIST-data/tfrecords/train.tfrecord')
for idx in range(train_data_length):
    tfrecord_obj = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.train.images[idx].tostring()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.train.labels[idx].tostring()])),
 }))
    writer.write(tfrecord_obj.SerializeToString())
writer.close()

writer = tf.python_io.TFRecordWriter('./MNIST-data/tfrecords/test.tfrecord')
for idx in range(test_data_length):
    tfrecord_obj = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.test.images[idx].tostring()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mnist.test.labels[idx].tostring()])),
 }))
    writer.write(tfrecord_obj.SerializeToString())
writer.close()
