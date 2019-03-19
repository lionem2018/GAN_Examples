import os, time, itertools, imageio, pickle, io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, 4096, kernel_initializer=w_init)
        o = tf.nn.tanh(dense2)

        return o

# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        lrelu1 = lrelu(dense1, 0.2)

        dense2 = tf.layers.dense(lrelu1, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense2)

        return o, dense2

# label preprocess
img_size = 64
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

onehot = np.eye(2350)

temp_z_ = np.random.normal(0, 1, (10, 100))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((10, 1))

for i in range(2349):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
    temp = np.ones((10,1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)

fixed_y_ = onehot[fixed_y_.astype(np.int32)].squeeze()
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, y: fixed_y_, isTrain: False})

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


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
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH*IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return label, image


# training parameters
batch_size = 100
lr = 0.0002
train_epoch = 200
img_size = 64
input_size = 4096
z_size = 100

# Default paths.
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  'hangul_dataset/labels/2350-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'hangul_dataset/tfrecords-output')


'''# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
train_label = mnist.train.labels
'''
#load custom Dataset
#tf.reset_default_graph()

labels = io.open(DEFAULT_LABEL_FILE, 'r', encoding='utf-8').read().splitlines()
num_classes = len(labels)

print('Processing data...')

tf_record_pattern = os.path.join(DEFAULT_TFRECORDS_DIR, '%s-*' % 'train')
train_data_files = tf.gfile.Glob(tf_record_pattern)

"""
label, image = get_image(train_data_files, num_classes)

# Associate objects with a randomly selected batch of labels and images.
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size,
    capacity=2000,
    min_after_dequeue=1000)
"""

# Make tf.data.Dataset
# If you want to use one more parameter for decode, use 'lambda' for data.map
dataset = tf.data.TFRecordDataset(train_data_files)
dataset = dataset.map(lambda x: get_image(x, num_classes))
dataset = dataset.repeat(train_epoch)  # set epoch
dataset = dataset.shuffle(buffer_size=3 * batch_size)  # for getting data in each buffer size data part
dataset = dataset.batch(batch_size)  # set batch size

# Make iterator for dataset
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 4096))
y = tf.placeholder(tf.float32, shape=(None, 2350))
z = tf.placeholder(tf.float32, shape=(None, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Initialize the queue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)

# results save folder
root = 'Hangul_cGAN_results/'
model = 'Hangul_cGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# Initialize iterator for datset
sess.run(iterator.initializer)

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(503370 // batch_size): # training_images//batchsize

        # Get a random batch of images and labels from dataset
        train_labels, train_images = sess.run(next_element)

        #print('Got data!')
        # Get images, reshape and rescale to pass to D
        train_images = train_images.reshape((batch_size, 4096))
        train_images = train_images*2 - 1

        # update discriminator
        x_ = train_images
        y_ = train_labels

        z_ = np.random.normal(0, 1, (batch_size, 100))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, y: y_, z: z_, isTrain: True})
        D_losses.append(loss_d_)

        #print('training G!')
        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 100))
        y_ = np.random.randint(0, 2349, (batch_size, 1))
        y_ = onehot[y_.astype(np.int32)].squeeze()
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y: y_, isTrain: True})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

# Stop queue threads and close session.
coord.request_stop()
coord.join(threads)
sess.close()