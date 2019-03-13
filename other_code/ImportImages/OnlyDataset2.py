import os, sys
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt



def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)


def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image


def change_file_to_tfrecord(subset_dir, one_hot=True, sample_size=None):
    # Read trainval data
    filename_list = os.listdir(subset_dir)
    set_size = len(filename_list)

    path_ds = tf.data.Dataset.from_tensor_slices(filename_list)
    image_ds = path_ds.map(load_and_preprocess_image)

    print(type(image_ds))

    return image_ds

# load asirra image
# def change_file_to_tfrecord(subset_dir, one_hot=True, sample_size=None):
#     # Read trainval data
#     filename_list = os.listdir(subset_dir)
#     set_size = len(filename_list)
#
#     if sample_size is not None and sample_size < set_size:
#         # Randomly sample subset of data when sample_size is specified
#         filename_list = np.random.choice(filename_list, size=sample_size, replace=False)
#         set_size = sample_size
#     else:
#         # Just shuffle the filename list
#         np.random.shuffle(filename_list)
#
#     # Pre-allocate data arrays
#     X_set = np.empty((set_size, 256, 256, 3), dtype=np.float32)    # (N, H, W, 3)
#     y_set = np.empty((set_size), dtype=np.uint8)                   # (N,)
#
#     for i, filename in enumerate(filename_list):
#         if i % 1000 == 0:
#             print('Reading subset data: {}/{}...'.format(i, set_size), end='\r')
#         label = filename.split('.')[0]
#         if label == 'cat':
#             y = 0
#         else:  # label == 'dog'
#             y = 1
#         file_path = os.path.join(subset_dir, filename)
#         img = imread(file_path)    # shape: (H, W, 3), range: [0, 255]
#         img = resize(img, (256, 256), mode='constant').astype(np.float32)    # (256, 256, 3), [0.0, 1.0]
#
#         X_set[i] = img
#         y_set[i] = y
#
#     if one_hot:
#         # Convert labels to one-hot vectors, shape: (N, num_classes)
#         y_set_oh = np.zeros((set_size, 2), dtype=np.uint8)
#         y_set_oh[np.arange(set_size), y_set] = 1
#         y_set = y_set_oh
#
#     # Assume that each row of `features` corresponds to the same row as `labels`.
#     assert X_set.shape[0] == y_set.shape[0]
#
#     dataset = tf.data.Dataset.from_tensor_slices((X_set, y_set))
#
#     return dataset
#
#
# # Main
# n_epoch = 3
# batch_size = 2
#

tr_dataset = change_file_to_tfrecord('./asirra/train/', sample_size=100)

tr_iterator = tr_dataset.make_initializable_iterator()

X, y = tr_iterator.get_next()

with tf.Session() as sess:
    sess.run(tr_iterator.initializer)

    for i in range(100):
        X_data = sess.run(X)
        y_data = sess.run(y)

        if i % 10 == 0:
            plt.imshow(X_data.reshape(256, 256, 3), interpolation='nearest')
            plt.show()
