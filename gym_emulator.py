import numpy as np
import gym
import gym_ple
from scipy.misc import imresize, imsave
import random
from environment import BaseEnvironment, FramePool,ObservationPool
import logging
import sys
import json

# a changer si on veut jouer à d'autres jeux !!
IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2

class GymEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.game = args.game
        self.gym_env = gym.make(self.game)
        self.gym_env.reset()
        with open("gym_game_info.json", 'r') as d :
            data = json.load(d)
            self.game_info  = data[self.game]

        self.legal_actions = [i for i in range(self.gym_env.action_space.n)]
        self.screen_width = self.game_info["screen_width"]
        self.screen_height = self.game_info["screen_height"]

        self.random_start = args.random_start
        self.single_life_episodes = args.single_life_episodes
        self.call_on_new_frame = args.visualize
        self.global_step = 0

        # Processed historcal frames that will be fed in to the network
        # (i.e., four 84x84 images)
        self.rgb = args.rgb
        self.depth = 1
        if self.rgb : self.depth = 3
        #self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, NR_IMAGES), dtype=np.uint8), self.rgb)
        self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height, self.screen_width,1), dtype=np.uint8)
        #self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width), dtype=np.uint8),
        #                            self.__process_frame_pool)
        self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width, self.depth), dtype=np.uint8),
                                        self.__process_frame_pool)
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, self.depth, NR_IMAGES), dtype=np.uint8), self.rgb)


    def get_legal_actions(self):
        return self.legal_actions

    def rgb_to_gray(self, im):
        new_im = np.zeros((self.screen_height, self.screen_width,1))
        new_im[:,:,0] = 0.299 * im[:,:, 0] + 0.587 * im[:,:, 1] + 0.114 * im[:,:, 2]
        return new_im

    def __get_screen_image(self):
        """
        Get the current frame luminance
        :return: the current frame
        """
        im = self.gym_env.render(mode='rgb_array')
        print('SCREEN : '+str(im.shape))
        if self.rgb : self.rgb_screen = im
        else : self.gray_screen = self.rgb_to_gray(im)

        if self.call_on_new_frame:
            self.rgb_screen = im
            self.on_new_frame(self.rgb_screen)

        if self.rgb : return self.rgb_screen
        return self.gray_screen

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        print('new game')
        self.gym_env.reset()
        if self.random_start:
            print('random start')
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.gym_env.step(self.legal_actions[0])

    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool """
        img = np.amax(frame_pool, axis=0)
        if self.game_info["crop"] :
            img = img[:self.game_info["crop_height"], :self.game_info["crop_width"], :]
            if not self.rgb :
                img = np.reshape(img, (self.game_info["crop_height"], self.game_info["crop_width"]))
        else :
            if not self.rgb :
                img = np.reshape(img, (self.screen_height, self.screen_width))
        img = imresize(img, (84, 84), interp='nearest')
        img = img.astype(np.uint8)
        if not self.rgb :
            img = np.reshape(img, (84, 84, 1))
        return img

    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            obs, r, episode_over, info = self.gym_env.step(self.legal_actions[a])
            reward += r
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            obs, r, episode_over, info = self.gym_env.step(self.legal_actions[a])
            reward += r
            img = self.__get_screen_image()
            self.frame_pool.new_frame(img)
        return reward, episode_over

    def get_initial_state(self):
        """ Get the initial state """
        print('get initial state')
        self.__new_game()
        for step in range(NR_IMAGES):
            _ , episode_over = self.__action_repeat(0)
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if episode_over :
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        print('next')
        reward, episode_over = self.__action_repeat(np.argmax(action))
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        observation = self.observation_pool.get_pooled_observations()
        self.global_step += 1
        return observation, reward, episode_over

    def __is_terminal(self, episode_over):
        if episode_over :
            self.lives = self.gym_env.ale.lives()
        if self.single_life_episodes:
            return episode_over or (self.lives < self.max_lives)
        else:
            return over

    def __is_over(self):
        return self.gym_env_ale.game_over()

    def get_noop(self):
        return [1.0, 0.0]
