import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


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


def get_image(serialized_example, num_classes):
    """This method defines the retrieval image examples from TFRecords files.

    Here we will define how the images will be represented (grayscale,
    flattened, floating point arrays) and how labels will be represented
    (one-hot vectors).
    """

    # tf.data.Dataset.map() opens and reads files automatically
    # Just need decoding code for each TFRecord file
    """
    # Convert filenames to a queue for an input pipeline.
    file_queue = tf.train.string_input_producer(files)

    # Create object to read TFRecords.
    reader = tf.TFRecordReader()

    # Read the full set of features for a single example.
    key, example = reader.read(file_queue)
    """

    # Parse the example to get a dict mapping feature keys to tensors.
    # image/class/label: integer denoting the index in a classification layer.
    # image/encoded: string containing JPEG encoded image
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float64)
    image = tf.reshape(image, [IMAGE_WIDTH,IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes, dtype=np.uint8))
    
    return label, image


# i have changed the dtype of image and labels to avoid memory issue problems in this file 
# data_len = Get_dataset_length(self.train_data_files)
# print("Data length is ", data_len)
# data_len = np.ndarray(data_len, dtype=np.uint8)