import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from glob import glob
import scipy.misc
from random import shuffle

def RNN(x, weights, biases, n_input, n_steps, n_hidden):
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, axis=0)

    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])

def FC(x, n_in, n_out):
    w = tf.Variable(tf.random_uniform([n_in, n_out]))
    b = tf.Variable(tf.random_uniform([n_out]))

    out = tf.add(tf.matmul(x, w), b)
    out = tf.nn.relu(out)
    return out

def flatten(_input):
    shape = _input.get_shape().as_list()
    dim = shape[1]*shape[2]*shape[3]
    return tf.reshape(_input, [-1,dim])

def conv2d(name, _input, filters, size, channels, stride, padding = 'VALID'):
    w = conv_weight_variable([size,size, channels,filters], name + '_weights')
    b = conv_bias_variable([filters], size, size, channels, name + '_biases')
    conv = tf.nn.conv2d(_input, w, strides=[1, stride, stride, 1],
                        padding=padding, name=name + '_convs')
    out = tf.nn.relu(tf.add(conv, b), name='' + name + '_activations')
    return w, b, out

def max_pooling(name, _input, stride=None, padding='VALID'):
    shape = [1,2,2,1]
    return tf.nn.max_pool(_input, shape, strides=shape, padding = padding, name=name)

def conv_weight_variable(shape, name):
    w = shape[0]
    h = shape[1]
    input_channels = shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')

def conv_bias_variable(shape, w, h, input_channels, name):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name, dtype='float32')

def create_conv_net(x, img_size):
    input_conv = tf.scalar_mul(1.0/255.0, tf.cast(x, tf.float32))
    print(input_conv.shape)

    _, _, conv1 = conv2d('conv1', input_conv, 32, 5, 1, 1, padding = 'SAME')
    mp_conv1 = max_pooling('mp_conv1', conv1)
    _, _, conv2 = conv2d('conv2', mp_conv1, 32, 5, 32, 1, padding = 'SAME')
    mp_conv2 = max_pooling('mp_conv2', conv2)
    _, _, conv3 = conv2d('conv3', mp_conv2, 64, 4, 32, 1, padding = 'SAME')
    mp_conv3 = max_pooling('mp_conv3', conv3)
    _, _, conv4 = conv2d('conv4', mp_conv3, 64, 3, 64, 1, padding = 'SAME')

    fc5 = FC(flatten(conv4), 6400, 20)
    return fc5



def create_dataset(x, y, timesteps):
    n = len(x)
    dx, dy = [], []
    index = [i for i in range(n-timesteps)]
    for i in range(n-timesteps):
        dx.append(x[i:i+timesteps])
        dy.append(y[i+timesteps-1])
    shuffle(index)
    res_x = [dx[i] for i in index]
    res_y = [dy[i] for i in index]
    return np.array(res_x), np.array(res_y)


print('Action predictor')

# Parameters
learning_rate = 0.001
nb_epochs = 70
batch_size = 32
display_step = 10

# Network Parameters
img_size = 84
n_input = 84*84  # input is image
n_steps = 5  # timesteps
n_hidden = 128  # hidden layer num of features
n_out_lstm = 128
n_outputs = 20 #for seaquest, possible actions
LSTM_bool = False

path_dataset = 'dataset/'
data = glob(path_dataset+'x/*')
images = [scipy.misc.imread(d, flatten=True) for d in data]
with open(path_dataset+'y.txt', 'r') as d :
    fichier_entier = d.read()
    lignes = fichier_entier.split('\n')
    nb_actions = [int(lignes[i][-1]) for i in range(len(lignes)-1)]
    actions = [np.eye(n_outputs)[nb_actions[i]] for i in range(len(nb_actions))]
data_x, data_y = create_dataset(images, actions, n_steps)
print(data_x.shape)
print(data_y.shape)


# tf Graph input
if LSTM_bool :
    x = tf.placeholder("float32", [None, n_steps, img_size, img_size])
else :
    x = tf.placeholder("float32", [None, img_size, img_size, 1])
y = tf.placeholder("float32", [None, n_outputs])

# Define weights
weights_lstm = {'out': tf.Variable(tf.random_normal([n_hidden, n_out_lstm]))}
biases_lstm = {'out': tf.Variable(tf.random_normal([n_out_lstm]))}
d = 1.0 / np.sqrt(n_outputs)
w_fc = tf.Variable(tf.random_uniform([n_out_lstm, n_outputs]))
b_fc = tf.Variable(tf.random_uniform([n_outputs], minval=-d, maxval=d))
w_soft = tf.Variable(tf.random_uniform([n_outputs, n_outputs], minval=-d, maxval=d))
b_soft = tf.Variable(tf.random_uniform([n_outputs], minval=-d, maxval=d))

print('Define Neural Network')

if LSTM_bool :
    output_lstm = RNN(x, weights_lstm, biases_lstm, n_input, n_steps, n_hidden)
    output_fc = FC(output_lstm, w_fc, b_fc)
else :
    output_fc = create_conv_net(x, img_size)

out_soft = tf.nn.softmax(tf.add(tf.matmul(output_fc, w_soft), b_soft))

# Define loss (Euclidean distance) and optimizer
print('Define loss and optimizer')
individual_losses = tf.reduce_sum(tf.squared_difference(out_soft, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    print('Start training')
    for e in range(nb_epochs):
        for i in range(int(len(data_x)/batch_size)):
            batch_x, batch_y = data_x[i*batch_size:(i+1)*batch_size], data_y[i*batch_size:(i+1)*batch_size]
            if LSTM_bool :
                batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            else :
                batch_x = np.array([np.amax(f, axis=0) for f in batch_x])
                batch_x = batch_x.reshape((batch_size, img_size, img_size, 1))
            batch_y = batch_y.reshape((batch_size, n_outputs))
            #out = sess.run(out_soft, feed_dict={x: batch_x})
            #print(out[0])
            #print(batch_y[0])
            #l = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            #print(l)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if i % display_step == 0:
                # Calculate batch loss
                loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(e)+' - '+str(i)+ ", Minibatch Loss= " +
                      "{:.6f}".format(loss_value))
    print("Optimization Finished!")
