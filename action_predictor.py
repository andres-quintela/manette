import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from glob import glob
import scipy.misc
from random import shuffle

def RNN(x, weights, biases, n_input, n_steps, n_hidden):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, axis=0)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])

def FC(x, w, b):
    out = tf.add(tf.matmul(x, w), b)
    out = tf.nn.relu(out)
    return out

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
nb_epochs = 10
batch_size = 32
display_step = 2

# Network Parameters
n_input = 84*84  # input is image
n_steps = 5  # timesteps
n_hidden = 128  # hidden layer num of features
n_out_lstm = 128
n_outputs = 20 #for seaquest, possible actions

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
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_outputs])

# Define weights
weights_lstm = {'out': tf.Variable(tf.random_normal([n_hidden, n_out_lstm]))}
biases_lstm = {'out': tf.Variable(tf.random_normal([n_out_lstm]))}
w_fc = tf.Variable(tf.random_uniform([n_out_lstm, n_outputs]))
b_fc = tf.Variable(tf.random_uniform([n_outputs]))
w_soft = tf.Variable(tf.random_uniform([n_outputs, n_outputs]))
b_soft = tf.Variable(tf.random_uniform([n_outputs]))

print('Define Neural Network')
output_lstm = RNN(x, weights_lstm, biases_lstm, n_input, n_steps, n_hidden)
output_fc = FC(output_lstm, w_fc, b_fc)
out_soft = tf.nn.log_softmax(tf.add(tf.matmul(output_fc, w_soft), b_soft))

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
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            batch_y = batch_y.reshape((batch_size, n_outputs))

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if i % display_step == 0:
                # Calculate batch loss
                loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(e)+' - '+str(i)+ ", Minibatch Loss= " +
                      "{:.6f}".format(loss_value))
    print("Optimization Finished!")
