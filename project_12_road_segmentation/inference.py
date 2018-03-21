import tensorflow as tf
from os.path import join, expanduser
from src.train import load_vgg, layers
from main import optimize
from helper import save_inference_samples

data_dir = join(expanduser("~"), 'Workspace_priv', 'self-driving-car', 'project_12_road_segmentation', 'data')
runs_dir = join(expanduser("~"), 'Workspace_priv', 'self-driving-car', 'road_segmentation_prediction')

num_classes = 2
vgg_path = join(data_dir, 'vgg')
image_h, image_w = (160, 576)

sess = tf.Session()

# Load VGG pretrained
image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

# Add skip connections
output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

# Define placeholders
labels = tf.placeholder(tf.float32, shape=[None, image_h, image_w, num_classes])
learning_rate = tf.placeholder(tf.float32, shape=[])
logits, train_op, cross_entropy_loss = optimize(output, labels, learning_rate, num_classes)

init = tf.global_variables_initializer()
sess.run(init)

new_saver = tf.train.import_meta_graph('model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))

save_inference_samples(runs_dir, data_dir, sess, (image_h, image_w), logits, keep_prob, image_input)

