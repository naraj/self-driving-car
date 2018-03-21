import os
import argparse
import warnings
import tensorflow as tf
from src.helper import gen_batch_function, save_inference_samples
from distutils.version import LooseVersion
from os.path import join, expanduser
import src.project_tests as tests
from src.image_augmentation import perform_augmentation
from src.train import load_vgg, layers

# Check TensorFlow Version
from pip._vendor.distlib._backport import tarfile

assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
        tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn(
        'No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def maybe_download_and_extract(data_url):
    """Download and extract model tar file.

    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.

    Args:
      data_url: Web location of the tar file containing the pretrained model.
    """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(
                                  total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                        'bytes.')
        print('Extracting file from ', filepath)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    else:
        print(
            'Not extracting or downloading files, model already present in disk')


def optimize(net_prediction, labels, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param net_prediction: TF Tensor of the last layer in the neural network
    :param labels: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Unroll
    logits_flat = tf.reshape(net_prediction, (-1, num_classes))
    labels_flat = tf.reshape(labels, (-1, num_classes))

    # Define loss
    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels_flat,
                                                logits=logits_flat))

    # Define optimization step
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cross_entropy_loss)

    return logits_flat, train_step, cross_entropy_loss


def train_nn(sess, training_epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss,
             image_input, labels, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param training_epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param image_input: TF Placeholder for input images
    :param labels: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    # Variable initialization
    sess.run(tf.global_variables_initializer())

    lr = args.learning_rate

    for e in range(0, training_epochs):
        print("Epoch {:02d}".format(e))

        loss_this_epoch = 0.0

        for i in range(0, args.batches_per_epoch):
            print(" Batch {:03d}".format(i))

            # Load a batch of examples
            batch_x, batch_y = next(get_batches_fn(batch_size))
            if args.augmentation:
                batch_x, batch_y = perform_augmentation(batch_x, batch_y)

            _, cur_loss = sess.run(fetches=[train_op, cross_entropy_loss],
                                   feed_dict={image_input: batch_x,
                                              labels: batch_y, keep_prob: 0.25,
                                              learning_rate: lr})

            loss_this_epoch += cur_loss

        print('Epoch: {:02d}  -  Loss: {:.03f}'.format(e,
                                                       loss_this_epoch / args.batches_per_epoch))


def perform_tests():
    tests.test_for_kitti_dataset(data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)


def run():
    num_classes = 2

    image_h, image_w = (160, 576)

    with tf.Session() as sess:
        print("Training")

        # Path to vgg model
        vgg_path = join(data_dir, 'vgg')

        # Create function to get batches
        batch_generator = gen_batch_function(
            join(data_dir, 'data_road/training'), (image_h, image_w))

        # Load VGG pretrained
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(
            sess, vgg_path)

        # Add skip connections
        output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out,
                        num_classes)

        # Define placeholders
        labels = tf.placeholder(tf.float32,
                                shape=[None, image_h, image_w, num_classes])
        learning_rate = tf.placeholder(tf.float32, shape=[])

        logits, train_op, cross_entropy_loss = optimize(output, labels,
                                                        learning_rate,
                                                        num_classes)

        # Training parameters
        train_nn(sess, args.training_epochs, args.batch_size, batch_generator,
                 train_op, cross_entropy_loss,
                 image_input, labels, keep_prob, learning_rate)

        save_inference_samples(runs_dir, data_dir, sess, (image_h, image_w),
                               logits, keep_prob, image_input)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        save_path = saver.save(sess, "model.ckpt")
        print("Model saved in path: %s" % save_path)

class Args:
    def __init__(self):
        self.batch_size = 8
        self.batches_per_epoch = 100
        self.training_epochs = 5
        self.learning_rate = 1e-4
        self.augmentation = False
        self.gpu = 1


args = Args()
data_dir = './data/'
runs_dir = './road_segmentation_prediction/'

# Restore appropriate augmentation value
run()
