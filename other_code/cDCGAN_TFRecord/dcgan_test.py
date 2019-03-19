import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
from generator import Generator 
from discriminator import Discriminator
from ops import *
import os, time, random, io
import pickle as pkl
from random import randint

class DCGAN:
    def __init__(self, img_shape, epochs=50000, lr_gen=0.0001, lr_disc=0.0001, z_shape=100, num_classes = 256, batch_size=100, beta1=0.5, epochs_for_sample=500):
        
       
        self.rows, self.cols, self.channels = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_shape = z_shape
        self.num_classes = num_classes
        self.epochs_for_sample = epochs_for_sample
        self.generator = Generator(self.z_shape,self.num_classes, img_shape, self.batch_size)
        self.discriminator = Discriminator(self.channels, self.num_classes, img_shape)
        self.samples = []
        self.losses = []

        self.SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

        # Default paths.
        self.DEFAULT_LABEL_FILE = os.path.join(self.SCRIPT_PATH, './labels/256-common-hangul.txt')
        self.DEFAULT_TFRECORDS_DIR = os.path.join(self.SCRIPT_PATH, 'tfrecords-output')


        """Perform graph definition and model training.

        Here we will first create our input pipeline for reading in TFRecords
        files and producing random batches of images and labels.
        """

        labels = io.open(self.DEFAULT_LABEL_FILE, 'r', encoding='utf-8').read().splitlines()
        num_classes = len(labels)

        print('Processing data...')

        tf_record_pattern = os.path.join(self.DEFAULT_TFRECORDS_DIR, '%s-*' % 'train')
        self.train_data_files = tf.gfile.Glob(tf_record_pattern)

        """
        label, image = get_image(self.train_data_files, num_classes)

        # Associate objects with a randomly selected batch of labels and images.
        self.image_batch, self.label_batch = tf.train.shuffle_batch(
            [image, label], batch_size=self.batch_size,
            capacity=2000,
            min_after_dequeue=1000)
        """

        # Make tf.data.Dataset
        # If you want to use one more parameter for decode, use 'lambda' for data.map
        dataset = tf.data.TFRecordDataset(self.train_data_files)
        dataset = dataset.map(lambda x: get_image(x, self.num_classes))
        dataset = dataset.repeat(self.train_epoch)  # set epoch
        dataset = dataset.shuffle(buffer_size=3 * self.batch_size)  # for getting data in each buffer size data part
        dataset = dataset.batch(self.batch_size)  # set batch size
        dataset = dataset.prefetch(buffer_size=1)  # reduce GPU starvation

        # Make iterator for dataset
        self.iterator = dataset.make_initializable_iterator()
        self.next_element = self.iterator.get_next()

        self.phX = tf.placeholder(tf.float32, [None, self.rows, self.cols, self.channels])
        self.phZ = tf.placeholder(tf.float32, [None, self.z_shape])
        self.phY_g = tf.placeholder(tf.float32, [None, self.num_classes])
        self.phY_d = tf.placeholder(tf.float32, shape=(None,  self.rows, self.cols, self.num_classes))
    
        self.gen_out = self.generator.forward(self.phZ, self.phY_g) #output shape of this z is (?, 28, 28, 1)

        disc_logits_fake = self.discriminator.forward(self.gen_out, self.phY_d ) #out put shape of this logit is (?, 1)
        disc_logits_real = self.discriminator.forward(self.phX, self.phY_d ) # out put shape of this logit is (?, 1)
        
        disc_fake_loss = cost(tf.zeros_like(disc_logits_fake), disc_logits_fake)
        disc_real_loss = cost(tf.ones_like(disc_logits_real), disc_logits_real)

        self.disc_loss = tf.add(disc_fake_loss, disc_real_loss)
        self.gen_loss = cost(tf.ones_like(disc_logits_fake), disc_logits_fake)

        train_vars = tf.trainable_variables()

        self.disc_vars = [var for var in train_vars if 'd' in var.name]
        self.gen_vars = [var for var in train_vars if 'g' in var.name]

        self.disc_train = tf.train.AdamOptimizer(lr_disc,beta1=beta1).minimize(self.disc_loss, var_list=self.disc_vars)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1=beta1).minimize(self.gen_loss, var_list=self.gen_vars)
        


    def train(self):
        init = [tf.global_variables_initializer(), self.iterator.initializer]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(init)

        # Initialize the queue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        epoch_start_time = time.time()
        for i in range(self.epochs):
            # Get a random batch of images and labels.
            train_labels, train_images = self.sess.run(self.next_element)

            # Real image input for Real Discriminator,
            # Get images, reshape and rescale to pass to D
            batch_X = train_images.reshape((self.batch_size, self.rows, self.cols, self.channels))
            batch_X = batch_X * 2 - 1

            # Z noise for Generator
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape)) # Shape is [?, 100]

            # Label input for Generator
            batch_Y_g = train_labels
            batch_Y_g = batch_Y_g.reshape([self.batch_size, self.num_classes])

            # Label input for Discriminator
            batch_Y_d = train_labels    
            batch_Y_d = batch_Y_d.reshape([self.batch_size,1,1,self.num_classes])
            batch_Y_d = batch_Y_d * np.ones([self.batch_size, self.rows, self.cols, self.num_classes])

            _, d_loss = self.sess.run([self.disc_train, self.disc_loss], feed_dict={self.phX:batch_X, self.phZ:batch_Z, self.phY_g:batch_Y_g, self.phY_d:batch_Y_d})
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
            _, g_loss = self.sess.run([self.gen_train, self.gen_loss], feed_dict={self.phX:batch_X, self.phZ:batch_Z, self.phY_g:batch_Y_g, self.phY_d:batch_Y_d})
            
            if i % self.epochs_for_sample == 0:
                epoch_end_time = time.time()
                per_epoch_ptime = epoch_end_time - epoch_start_time

                print(f"Epoch: {i}. Discriminator loss: {d_loss}. Generator loss: {g_loss}")
                # Save losses to view after training
                self.losses.append((d_loss, g_loss))

        # Save training generator samples
        with open('train_samples.pkl', 'wb') as f:
            pkl.dump(self.samples, f)

        # Generate random sample after training
        self.generate_random_sample()
        
        # Stop queue threads and close session.
        coord.request_stop()
        coord.join(threads)
        self.sess.close() 



    def generate_random_sample(self):
        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(init)
        # Only save generator variables
        saver = tf.train.Saver(var_list=self.gen_vars)
        c = 7
        r = 7
        # data_len = Get_dataset_length(self.train_data_files)
        # data_len_y = np.ndarray(data_len, dtype=np.uint8)

        # z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
        # idx = np.random.randint(0, data_len, self.batch_size)
        # print('length of images are ', data_len)
        # print('Batch size is ', self.batch_size)
        # print('idx shape is is ', idx.shape)
        # print('Y shape is ', data_len_y.shape)
        
        # # Label input for Generator
        # batch_Y_g = np.eye(self.num_classes)[data_len_y]
        # batch_Y_g = batch_Y_g[idx]
        # batch_Y_g = batch_Y_g.reshape([self.batch_size, self.num_classes])
        n_sample = 100
        z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))

        # Create conditional one-hot vector, with index 5 = 1
        batch_Y_g = np.zeros(shape=[n_sample, 256])
        batch_Y_g[:, 0] = 4
        saver.restore(self.sess, tf.train.latest_checkpoint('checkpoints'))
        samples = self.sess.run(self.gen_out, feed_dict={self.phZ:z, self.phY_g:batch_Y_g})

        # scale between 0, 1
        fig, axs = plt.subplots(c, r)
        cnt = 0
        for i in range(c):
            for j in range(r):
                axs[i, j].imshow(samples[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("generated/generated_test_1.png")
        plt.close()


if __name__ == '__main__':
    img_shape = (64, 64, 1)
    epochs = 50000
    dcgan = DCGAN(img_shape, epochs)

    if not os.path.exists('samples/'):
        os.makedirs('samples/')
    
    dcgan.generate_random_sample()