import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn
from networks import Operations
from networks import PpwwyyxxNetwork




with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./logs/pong/checkpoints/-1000000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs/pong/checkpoints/'))

    vars = tf.trainable_variables()
    print(vars) #some infos about variables...
    vars_vals = sess.run(vars)
    for var, val in zip(vars, vars_vals):
        print("var: {}, value: {}".format(var.name, val)) #...or sort it in a list....
