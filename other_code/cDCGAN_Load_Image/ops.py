import numpy as np
import tensorflow as tf
import os, io, random

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_LABEL_CSV = os.path.join("~/Desktop/sslab-deeplearning/GAN models/cGAN_cDCGAN_TF/cDCGAN_Hangul_OOP/image-data/labels-map.csv")
DEFAULT_LABEL_FILE = os.path.join("~/Desktop/sslab-deeplearning/GAN models/cGAN_cDCGAN_TF/cDCGAN_Hangul_OOP/labels/256-common-hangul.txt")

DEFAULT_NUM_SHARDS_TRAIN = 3
DEFAULT_NUM_SHARDS_TEST = 1

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.02))


def init_bias(shape):
    return tf.Variable(tf.zeros(shape))


def conv2d(x, filter, strides, padding):
    return tf.nn.conv2d(x, filter, strides=strides, padding=padding)


def cost(labels, logits):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))


def Get_dataset_length(train_data_files):
    c = 0
    for fn in train_data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def get_image_path_and_labels():

    labels_csv = DEFAULT_LABEL_CSV
    label_file = DEFAULT_LABEL_FILE

    labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
    labels_file = io.open(label_file, 'r', encoding='utf-8').read().splitlines()

    label_dict = {}
    count = 0
    for label in labels_file:
        label_dict[label] = count
        count += 1

    images = []
    labels = []
    for row in labels_csv:
        file, label = row.strip().split(',')
        images.append(file)
        labels.append(label_dict[label])

    shuffled_indices = list(range(len(images)))
    random.seed(12121)
    random.shuffle(shuffled_indices)
    filenames = [images[i] for i in shuffled_indices]
    labels = [labels[i] for i in shuffled_indices]

    return filenames, labels


def make_dataset(filenames, labels, num_classes):

    images = []
    one_hot_labels = []

    for filename, label in [filenames, labels]:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            im_data = f.read()
            images.append(tf.image.decode_jpeg(im_data, channels=1))

            label = np.zeros((1, num_classes), dtype=np.uint8)
            label[1, label] = 1
            one_hot_labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((images, one_hot_labels))

    return dataset
