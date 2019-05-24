import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import tensorflow as tf
from tensorflow.contrib import rnn

from collections import OrderedDict
from scipy.misc import imresize, imsave
import gym


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=(8,8), stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(4,4), stride=2)
        self.fc3 = nn.Linear(2592, 256)
        self.output = nn.Linear(256, 6)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc3(torch.flatten(x).view(1,2592)))
        return self.output(x.view(x.size(0), -1))





with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./logs/spaceinvaders/checkpoints/-19000000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs/spaceinvaders/checkpoints/'))

    vars = tf.trainable_variables()
    #print(vars) #some infos about variables...
    vars_vals = sess.run(vars)
    for var, val in zip(vars, vars_vals):
        #print("var: {}, value: {}".format(var.name, val))
        print("var: {}".format(var.name))


    conv1_weights = [v for v in tf.trainable_variables() if v.name == "Network/conv1/conv1_weights:0"][0]
    conv1_weights_val=sess.run(conv1_weights).transpose()
    conv1_w=torch.from_numpy(conv1_weights_val)


    conv1_bias = [v for v in tf.trainable_variables() if v.name == "Network/conv1/conv1_biases:0"][0]
    conv1_bias_val=sess.run(conv1_bias).transpose()
    conv1_b=torch.from_numpy(conv1_bias_val)



    conv2_weights = [v for v in tf.trainable_variables() if v.name == "Network/conv2/conv2_weights:0"][0]
    conv2_weights_val=sess.run(conv2_weights).transpose()
    conv2_w=torch.from_numpy(conv2_weights_val)


    conv2_bias = [v for v in tf.trainable_variables() if v.name == "Network/conv2/conv2_biases:0"][0]
    conv2_bias_val=sess.run(conv2_bias).transpose()
    conv2_b=torch.from_numpy(conv2_bias_val)


    fc3_weights = [v for v in tf.trainable_variables() if v.name == "Network/fc3/fc3_weights:0"][0]
    fc3_weights_val=sess.run(fc3_weights).transpose()
    fc3_w=torch.from_numpy(fc3_weights_val)

    fc3_bias = [v for v in tf.trainable_variables() if v.name == "Network/fc3/fc3_biases:0"][0]
    fc3_bias_val=sess.run(fc3_bias).transpose()
    fc3_b=torch.from_numpy(fc3_bias_val)

    output_weights = [v for v in tf.trainable_variables() if v.name == "Training/Actor/actor_output/actor_output_weights:0"][0]
    output_weights_val=sess.run(output_weights).transpose()
    output_w=torch.from_numpy(output_weights_val)

    output_bias = [v for v in tf.trainable_variables() if v.name == "Training/Actor/actor_output/actor_output_biases:0"][0]
    output_bias_val=sess.run(output_bias).transpose()
    output_b=torch.from_numpy(output_bias_val)



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


def rgb_to_gray(im):
    new_im = np.zeros((210,160,1))
    new_im[:,:,0] = 0.299 * im[:,:, 0] + 0.587 * im[:,:, 1] + 0.114 * im[:,:, 2]
    return new_im


def get_screen(state):

    img = rgb_to_gray(state)

    img = np.reshape(img, (210, 160))
    img = imresize(img, (84, 84), interp='nearest')
    img = img.astype(np.uint8)
    img = np.reshape(img, (84, 84, 1))
    img = torch.tensor([img]).float()
    return img

# state=get_screen(env)
#
# state_image=np.concatenate([state,state,state,state],axis=-1)
# state=torch.tensor([state_image])
# state=state.permute(0,3,1,2).float()
# print(state.shape)
#
#
#
# network(state)

def get_state(next_state):
    state=get_screen(next_state)
    state_image=np.concatenate([state,state,state,state],axis=-1)
    state_image=torch.tensor(state_image)
    state_image=state_image.permute(0,3,1,2).float()
    return state_image


def state_from_list(frame_list):
    state_image=np.concatenate([frame_list[0],frame_list[1],frame_list[2],frame_list[3]],axis=-1)
    state_image=torch.tensor(state_image)
    state_image=state_image.permute(0,3,1,2).float()
    return state_image


def get_next_state(state,next_state_image):

    next_state_tensor=get_screen(next_state_image).permute(0,3,1,2)
    next_state_stack=torch.cat((next_state_tensor,state),1)
    next_state_stack=next_state_stack[:,:4,:,:]
    return next_state_stack


from gym.envs.atari.atari_env import AtariEnv
#env1 = AtariEnv(game='space_invaders', obs_type='image', frameskip=1, full_action_space=False)
env1=gym.make("SpaceInvaders-v0")

next_state=env1.reset()
state=get_state(next_state)
for t in range(0,100000):
    env1.render()
    actions=network(state)
    next_state_image,r,_,done=env1.step(actions.max(1)[1])
    state=get_next_state(state,next_state_image)
    print(state)
