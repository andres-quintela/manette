import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from glob import glob
import scipy.misc
from random import shuffle
import matplotlib.pyplot as plt
from networks import Operations

print('---- Action predictor ----')

# Parameters
learning_rate = 0.001
nb_epochs = 5
batch_size = 32
display_step = 5

# Network Parameters
img_size = 84
n_input = 640  # same as output conv net
n_steps = 5  # timesteps
n_hidden = 128  # hidden layer num of features
n_out_lstm = 128
n_outputs = 20 #for seaquest, possible actions
LSTM_bool = False

op = Operations({'rgb':False, 'alpha_leaky_relu':0.1})

def RNN(x, weights, biases, n_input, n_steps, n_hidden):
    print('RNN...')
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
    return tf.nn.bias_add(tf.matmul(outputs[-1], weights), biases)

def create_conv_net(x, img_size, n_outputs):
    with tf.name_scope('net'):
        input_conv = tf.scalar_mul(1.0/255.0, tf.cast(x, tf.float32))
        print(input_conv.shape)

        _, _, conv1 = op.conv2d('conv1', input_conv, 32, 5, 1, 1, padding = 'SAME')
        mp_conv1 = op.max_pooling('mp_conv1', conv1)
        _, _, conv2 = op.conv2d('conv2', mp_conv1, 32, 5, 32, 1, padding = 'SAME')
        mp_conv2 = op.max_pooling('mp_conv2', conv2)
        _, _, conv3 = op.conv2d('conv3', mp_conv2, 64, 4, 32, 1, padding = 'SAME')
        mp_conv3 = op.max_pooling('mp_conv3', conv3)
        _, _, conv4 = op.conv2d('conv4', mp_conv3, 64, 3, 64, 1, padding = 'SAME')

        print('conv 4 : '+str(conv4.shape))
        _, _, fc5 = op.fc('fc5', op.flatten(conv4), n_outputs)

    print('fc5 : '+str(fc5.shape))
    return fc5

def create_lstm_net(x, img_size, n_input, n_steps, n_hidden, n_outputs):
    with tf.name_scope('net'):
        x_img = tf.reshape(x, [-1, img_size, img_size, 1])
        print('x_img : '+str(x_img.shape))

        # probleme ordre des images ?
        out_conv = create_conv_net(x_img, img_size, n_input)
        print('out_conv : '+str(out_conv.shape))

        w_lstm = tf.Variable(tf.random_normal([n_hidden, n_hidden]))
        b_lstm = tf.Variable(tf.random_normal([n_hidden]))

        x_lstm = tf.reshape(out_conv, [-1, n_steps, n_input])
        print('x_lstm : '+str(x_lstm.shape))
        out_lstm = RNN(x_lstm, w_lstm, b_lstm, n_input, n_steps, n_hidden)

        _, _, fc6 = op.fc('fc6', out_lstm, n_outputs)
    return fc6

def create_dataset(timesteps, n_outputs):
    path_dataset = 'dataset/'
    data = glob(path_dataset+'x/*')
    images = [scipy.misc.imread(d, flatten=True) for d in data]
    with open(path_dataset+'y.txt', 'r') as d :
        fichier_entier = d.read()
        lignes = fichier_entier.split('\n')
        nb_actions = [int(lignes[i][-1]) for i in range(len(lignes)-1)]
        actions = [np.eye(n_outputs)[nb_actions[i]] for i in range(len(nb_actions))]
    n = len(images)
    dx, dy = [], []
    index = [i for i in range(n-timesteps)]
    for i in range(n-timesteps):
        dx.append(images[i:i+timesteps])
        dy.append(actions[i+timesteps-1])
    shuffle(index)
    res_x = [dx[i] for i in index]
    res_y = [dy[i] for i in index]
    return np.array(res_x), np.array(res_y)


# import dataset
data_x, data_y = create_dataset(n_steps, n_outputs)
print(data_x.shape)
print(data_y.shape)

# tf Graph input
if LSTM_bool : x = tf.placeholder("float32", [None, n_steps, img_size, img_size])
else : x = tf.placeholder("float32", [None, img_size, img_size, 1])
y = tf.placeholder("float32", [None, n_outputs])

if LSTM_bool : output_fc = create_lstm_net(x, img_size, n_input, n_steps, n_hidden, n_outputs)
else : output_fc = create_conv_net(x, img_size, n_outputs)

# Define weights for softmax
d = 1.0 / np.sqrt(n_outputs)
w_soft = tf.Variable(tf.random_uniform([n_outputs, n_outputs], minval=-d, maxval=d))
b_soft = tf.Variable(tf.random_uniform([n_outputs], minval=-d, maxval=d))

out_soft = tf.nn.softmax(tf.add(tf.matmul(output_fc, w_soft), b_soft))
predictions = tf.argmax(out_soft, axis=1)
labels = tf.argmax(y, axis=1)

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(out_soft, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
accuracy = tf.metrics.accuracy(labels, predictions)

# Initializing the variables
init = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    sess.run(init_l)
    # Keep training until reach max iterations
    print('-- Start training ...')
    total_batch = 0
    plot_X = []
    plot_Y = []
    plot_Z = []

    for e in range(nb_epochs):
        for i in range(int(len(data_x)/batch_size)):
            batch_x, batch_y = data_x[i*batch_size:(i+1)*batch_size], data_y[i*batch_size:(i+1)*batch_size]
            if not LSTM_bool :
                batch_x = np.array([np.amax(f, axis=0) for f in batch_x])
                batch_x = batch_x.reshape((batch_size, img_size, img_size, 1))
            batch_y = batch_y.reshape((batch_size, n_outputs))
            out = sess.run(out_soft, feed_dict={x: batch_x})
            pred = sess.run(predictions, feed_dict={x: batch_x})
            #print(out[0])
            #print(pred[0])
            #lab = sess.run(labels, feed_dict={y: batch_y})
            #print(lab[0])
            #l = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            #print(l)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if i % display_step == 0:
                # Calculate batch loss
                loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                acc_value = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                print("Iter "+str(e)+' - '+str(i)+", Minibatch Loss= "+str(loss_value)+" , accuracy = "+str(acc_value))
                plot_X.append(total_batch)
                plot_Y.append(loss_value)
                plot_Z.append(acc_value[0])
            total_batch += 1
    print("-- Optimization Finished!")
    plt.plot(plot_X, plot_Y)
    #plt.savefig('dataset/loss-lstm'+str(LSTM_bool)+'-in'+str(n_input)+'-hidden'+str(n_hidden)+'.png')
    plt.plot(plot_X, plot_Z)
    plt.savefig('dataset/acc-lstm'+str(LSTM_bool)+'-in'+str(n_input)+'-hidden'+str(n_hidden)+'.png')
