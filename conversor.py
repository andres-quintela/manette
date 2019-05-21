import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
from networks import Operations
from networks import PpwwyyxxNetwork





with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./logs/seaquest/checkpoints/-3000000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs/seaquest/checkpoints/'))
