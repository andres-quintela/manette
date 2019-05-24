import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from tensorflow.contrib import rnn

from collections import OrderedDict
from scipy.misc import imresize, imsave
import gym




import os
from train import get_network_and_environment_creator
import logger_utils
import argparse
import numpy as np
import time
import tensorflow as tf
import random
from paac import PAACLearner
from exploration_policy import ExplorationPolicy, Action

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4,4), stride=2)
        self.fc3 = nn.Linear(2592, 256)
        self.output = nn.Linear(256, 6)



    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.fc3(torch.flatten(x).view(1,2592)))
        return self.output(x.view(x.size(0), -1))





with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./logs/pong/checkpoints/-14000000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs/pong/checkpoints/'))

    vars = tf.trainable_variables()
    #print(vars) #some infos about variables...
    vars_vals = sess.run(vars)
    # for var, val in zip(vars, vars_vals):
    #     #print("var: {}, value: {}".format(var.name, val))
    #     print("var: {}".format(var.name))


    conv1_weights = [v for v in tf.trainable_variables() if v.name == "Network/conv1/conv1_weights:0"][0]
    conv1_weights_val=sess.run(conv1_weights)
    conv1_w=torch.from_numpy(conv1_weights_val).permute(3,2,0,1)


    conv1_bias = [v for v in tf.trainable_variables() if v.name == "Network/conv1/conv1_biases:0"][0]
    conv1_bias_val=sess.run(conv1_bias)
    conv1_b=torch.from_numpy(conv1_bias_val)
    #conv1_b=torch.flip(conv1_b,[0])


    conv2_weights = [v for v in tf.trainable_variables() if v.name == "Network/conv2/conv2_weights:0"][0]
    conv2_weights_val=sess.run(conv2_weights)
    conv2_w=torch.from_numpy(conv2_weights_val).permute(3,2,0,1)


    conv2_bias = [v for v in tf.trainable_variables() if v.name == "Network/conv2/conv2_biases:0"][0]
    conv2_bias_val=sess.run(conv2_bias)
    conv2_b=torch.from_numpy(conv2_bias_val)
    #conv2_b=torch.flip(conv2_b,[0])


    fc3_weights = [v for v in tf.trainable_variables() if v.name == "Network/fc3/fc3_weights:0"][0]
    fc3_weights_val=sess.run(fc3_weights)
    fc3_w=torch.from_numpy(fc3_weights_val).permute(1,0)

    fc3_bias = [v for v in tf.trainable_variables() if v.name == "Network/fc3/fc3_biases:0"][0]
    fc3_bias_val=sess.run(fc3_bias)
    fc3_b=torch.from_numpy(fc3_bias_val)
    #fc3_b=torch.flip(fc3_b,[0])

    output_weights = [v for v in tf.trainable_variables() if v.name == "Training/Actor/actor_output/actor_output_weights:0"][0]
    output_weights_val=sess.run(output_weights)
    output_w=torch.from_numpy(output_weights_val).permute(1,0)

    output_bias = [v for v in tf.trainable_variables() if v.name == "Training/Actor/actor_output/actor_output_biases:0"][0]
    output_bias_val=sess.run(output_bias)
    output_b=torch.from_numpy(output_bias_val)
    #output_b=torch.flip(output_b,[0])



network=DQN()

#import parameters

new_state_dict = OrderedDict({'conv1.weight': conv1_w})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'conv1.bias': conv1_b})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'conv2.weight': conv2_w})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'conv2.bias': conv2_b})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'fc3.weight': fc3_w})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'fc3.bias': fc3_b})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'output.weight': output_w})
network.load_state_dict(new_state_dict, strict=False)

new_state_dict = OrderedDict({'output.bias': output_b})
network.load_state_dict(new_state_dict, strict=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where to save the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-gn', '--gif_name', default=None, type=str, help="If provided, a gif will be produced and stored with this name", dest="gif_name")
    parser.add_argument('-gf', '--gif_folder', default='', type=str, help="The folder where to save gifs.", dest="gif_folder")
    parser.add_argument('-d', '--device', default='/cpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")

    args = parser.parse_args()
    arg_file = os.path.join(args.folder, 'args.json')
    device = args.device
    for k, v in logger_utils.load_args(arg_file).items():
        setattr(args, k, v)
    args.max_global_steps = 0
    df = args.folder
    args.debugging_folder = '/tmp/logs'
    args.device = device

    explo_policy = ExplorationPolicy(args, test = False)
    _, env_creator = get_network_and_environment_creator(args, explo_policy)


    environment = env_creator.create_environment(1)

    def tf_to_py_state(states):
        state=torch.tensor([states])
        state=state.permute(0,3,1,2).float()
        return state


    initial_state = environment.get_initial_state()
    initial_state = tf_to_py_state(initial_state)

    reward=0
    state=initial_state
    episode_over=False
    while not episode_over:
        environment.gym_env.render()
        actions=network(state)
        state, r, episode_over=environment.next(actions.max(1)[1])
        state=tf_to_py_state(state)
        reward+=r
    print(reward)
