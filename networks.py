import tensorflow as tf
import logging
import numpy as np

class Operations():
    def __init__(self, conf):
        self.rgb = conf['rgb']
        self.depth = 1
        if self.rgb :
            self.depth = 3

    def flatten(self, _input):
        shape = _input.get_shape().as_list()
        dim = shape[1]*shape[2]*shape[3]
        return tf.reshape(_input, [-1,dim], name='_flattened')


    def conv2d(self, name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch"):
        w = self.conv_weight_variable([size,size, channels,filters],
                                 name + '_weights', init = init)
        b = self.conv_bias_variable([filters], size, size, channels,
                               name + '_biases', init = init)
        conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1],
                            padding=padding, name=name + '_convs')
        out = tf.nn.relu(tf.add(conv, b),
                         name='' + name + '_activations')
        return w, b, out


    def conv_weight_variable(self, shape, name, init = "torch"):
        if init == "glorot_uniform":
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
            d = np.sqrt(6. / (fan_in + fan_out))
        else:
            w = shape[0]
            h = shape[1]
            input_channels = shape[3]
            d = 1.0 / np.sqrt(input_channels * w * h)

        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')


    def conv_bias_variable(self, shape, w, h, input_channels, name, init= "torch"):
        if init == "glorot_uniform":
            initial = tf.zeros(shape)
        else:
            d = 1.0 / np.sqrt(input_channels * w * h)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')


    def fc(self, name, _input, output_dim, activation = "relu", init = "torch"):
        input_dim = _input.get_shape().as_list()[1]
        w = self.fc_weight_variable([input_dim, output_dim],
                               name + '_weights', init = init)
        b = self.fc_bias_variable([output_dim], input_dim,
                             '' + name + '_biases', init = init)
        out = tf.add(tf.matmul(_input, w), b, name= name + '_out')

        if activation == "relu":
            out = tf.nn.relu(out, name='' + name + '_relu')

        return w, b, out


    def fc_weight_variable(self, shape, name, init="torch"):
        if init == "glorot_uniform":
            fan_in = shape[0]
            fan_out = shape[1]
            d = np.sqrt(6. / (fan_in + fan_out))
        else:
            input_channels = shape[0]
            d = 1.0 / np.sqrt(input_channels)
        initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')


    def fc_bias_variable(self, shape, input_channels, name, init= "torch"):
        if init=="glorot_uniform":
            initial = tf.zeros(shape, dtype='float32')
        else:
            d = 1.0 / np.sqrt(input_channels)
            initial = tf.random_uniform(shape, minval=-d, maxval=d)
        return tf.Variable(initial, name=name, dtype='float32')


    def softmax(self, name, _input, output_dim, temp):
        softmax_temp = tf.constant(temp, dtype=tf.float32)
        input_dim = _input.get_shape().as_list()[1]
        w = self.fc_weight_variable([input_dim, output_dim], name + '_weights')
        b = self.fc_bias_variable([output_dim], input_dim, name + '_biases')
        out = tf.nn.softmax(tf.div(tf.add(tf.matmul(_input, w), b), softmax_temp), name= name + '_policy')
        return w, b, out


    def log_softmax(self, name, _input, output_dim):
        input_dim = _input.get_shape().as_list()[1]
        w = self.fc_weight_variable([input_dim, output_dim], name + '_weights')
        b = self.fc_bias_variable([output_dim], input_dim, name + '_biases')
        out = tf.nn.log_softmax(tf.add(tf.matmul(_input, w), b), name= name + '_policy')
        return w, b, out

    def max_pooling(self, name, _input, stride=None, padding='VALID'):
        shape = [1,2,2,1]
        return tf.nn.max_pool(_input, shape, strides=shape, padding = padding, name=name)


class Network(object):

    def __init__(self, conf):

        self.name = conf['name']
        self.num_actions = conf['num_actions']
        self.clip_norm = conf['clip_norm']
        self.clip_norm_type = conf['clip_norm_type']
        self.device = conf['device']
        self.rgb = conf['rgb']
        self.depth = 1
        if self.rgb :
            self.depth = 3
        self.op = Operations(conf)
        self.max_repetition = conf['max_repetition']
        self.total_repetitions = self.max_repetition + 1

        with tf.device(self.device):
            with tf.name_scope(self.name):
                self.loss_scaling = 5.0

                self.input_ph = tf.placeholder(tf.uint8, [None, 84, 84, self.depth* 4], name='input')
                self.selected_action_ph = tf.placeholder("float32", [None, self.num_actions], name="selected_action")
                self.selected_repetition_ph = tf.placeholder("float32", [None, self.total_repetitions], name="selected_repetition")
                self.input = tf.scalar_mul(1.0/255.0, tf.cast(self.input_ph, tf.float32))

                # This class should never be used, must be subclassed

                # The output layer
                self.output = None

    def init(self, checkpoint_folder, saver, session):
        last_saving_step = 0

        with tf.device('/cpu:0'):
            # Initialize network parameters
            path = tf.train.latest_checkpoint(checkpoint_folder)
            if path is None:
                logging.info('Initializing all variables')
                session.run(tf.global_variables_initializer())
            else:
                logging.info('Restoring network variables from previous run')
                saver.restore(session, path)
                last_saving_step = int(path[path.rindex('-')+1:])
        return last_saving_step


class NIPSNetwork(Network):

    def __init__(self, conf):
        super(NIPSNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                w_conv1, b_conv1, conv1 = self.op.conv2d('conv1', self.input, 16, 8, self.depth*4, 4)

                w_conv2, b_conv2, conv2 = self.op.conv2d('conv2', conv1, 32, 4, 16, 2)

                w_fc3, b_fc3, fc3 = self.op.fc('fc3', self.op.flatten(conv2), 256, activation="relu")

                tf.summary.histogram("w_conv1", w_conv1)
                tf.summary.histogram("w_conv2", w_conv2)
                tf.summary.histogram("b_conv1", b_conv1)
                tf.summary.histogram("b_conv2", b_conv2)

                self.output = fc3

class BayesianNetwork(NIPSNetwork):
    def __init__(self, conf):
        super(BayesianNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                logging.info('Using bayesion Network')
                dropout = tf.nn.dropout(self.output, conf["keep_percentage"])

                w_fc4, b_fc4, fc4 = fc('fc4', dropout, 256, activation="relu")

                self.output = fc4

class PpwwyyxxNetwork(Network):
    def __init__(self, conf):
        super(PpwwyyxxNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):

#conv2d(name, _input, filters, size, channels, stride, padding = 'VALID', init = "torch")

                _, _, conv1 = self.op.conv2d('conv1', self.input, 32, 5, self.depth * 4, 1, padding = 'SAME')
                mp_conv1 = self.op.max_pooling('mp_conv1', conv1)
                _, _, conv2 = self.op.conv2d('conv2', mp_conv1, 32, 5, 32, 1, padding = 'SAME')
                mp_conv2 = self.op.max_pooling('mp_conv2', conv2)
                _, _, conv3 = self.op.conv2d('conv3', mp_conv2, 64, 4, 32, 1, padding = 'SAME')
                mp_conv3 = self.op.max_pooling('mp_conv3', conv3)
                _, _, conv4 = self.op.conv2d('conv4', mp_conv3, 64, 3, 64, 1, padding = 'SAME')

                _, _, fc5 = self.op.fc('fc5', self.op.flatten(conv4), 512, activation="relu")

                self.output = fc5


class NatureNetwork(Network):

    def __init__(self, conf):
        super(NatureNetwork, self).__init__(conf)

        with tf.device(self.device):
            with tf.name_scope(self.name):
                _, _, conv1 = self.op.conv2d('conv1', self.input, 32, 8, self.depth*4, 4)

                _, _, conv2 = self.op.conv2d('conv2', conv1, 64, 4, 32, 2)

                _, _, conv3 = self.op.conv2d('conv3', conv2, 64, 3, 64, 1)

                _, _, fc4 = self.op.fc('fc4', self.op.flatten(conv3), 512, activation="relu")

                self.output = fc4
