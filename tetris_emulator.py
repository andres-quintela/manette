import numpy as np
from tetris import TetrisApp
from scipy.misc import imresize, imsave
import cv2
import random
from environment import BaseEnvironment, FramePool, ObservationPool
import logging

IMG_SIZE_X = 84
IMG_SIZE_Y = 84
NR_IMAGES = 4
ACTION_REPEAT = 4
MAX_START_WAIT = 30
FRAMES_IN_POOL = 2


class TetrisEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.tetris = TetrisApp(emulator = True)

        self.legal_actions = [0,1,2,3,4]
        self.screen_width, self.screen_height = 288, 396
        self.lives = 1

        self.random_start = args.random_start
        self.single_life_episodes = args.single_life_episodes
        self.call_on_new_frame = args.visualize
        self.global_step = 0

        self.compteur = 0

        # Processed historcal frames that will be fed in to the network
        # (i.e., four 84x84 images)
        self.rgb = args.rgb
        self.depth = 1
        if self.rgb : self.depth = 3
        self.rgb_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self.gray_screen = np.zeros((self.screen_height, self.screen_width,1), dtype=np.uint8)
        self.frame_pool = FramePool(np.empty((2, self.screen_height,self.screen_width, self.depth), dtype=np.uint8),
                                        self.__process_frame_pool)
        self.observation_pool = ObservationPool(np.zeros((IMG_SIZE_X, IMG_SIZE_Y, self.depth, NR_IMAGES), dtype=np.uint8), self.rgb)


    def get_legal_actions(self):
        return self.legal_actions

    def __get_screen_image(self):
        """ Get the current frame luminance. Return: the current frame """
        fucking_name = 'img'+str(self.compteur)+'.jpg'
        self.gray_screen = self.tetris.getScreen(fucking_name, rgb=False)
        if self.rgb :
            self.rgb_screen = self.tetris.getScreen(fucking_name)
        if self.call_on_new_frame:
            self.rgb_screen = self.tetris.getScreen(fucking_name)
            self.on_new_frame(self.rgb_screen)
        self.compteur += 1
        if self.rgb :
            return self.rgb_screen
        return self.gray_screen

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        self.tetris.init_game()
        self.lives = 1
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.tetris.act(0)

    def __process_frame_pool(self, frame_pool):
        """ Preprocess frame pool """
        img = np.amax(frame_pool, axis=0)
        if not self.rgb :
            img = np.reshape(img, (210, 160))
        img = imresize(img, (84, 84), interp='nearest')
        img = img.astype(np.uint8)
        if not self.rgb :
            img = np.reshape(img, (84, 84, 1))
        return img

    def __action_repeat(self, a, times=ACTION_REPEAT):
        """ Repeat action and grab screen into frame pool """
        reward = 0
        for i in range(times - FRAMES_IN_POOL):
            reward += self.tetris.act(a)
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            reward += self.tetris.act(a)
            img = self.__get_screen_image()
            # print('DEPTH : '+str(self.depth))
            # print('img : '+str(img.shape))
            # #truc = np.array(img[:,:,0])
            # imsave('actionrepeat.jpg', img)
            self.frame_pool.new_frame(img)
        return reward

    def get_initial_state(self):
        """ Get the initial state """
        self.__new_game()
        for step in range(NR_IMAGES):
            _ = self.__action_repeat(0)
            self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        if self.__is_terminal():
            raise Exception('This should never happen.')
        return self.observation_pool.get_pooled_observations()

    def next(self, action):
        """ Get the next state, reward, and game over signal """
        reward = self.__action_repeat(action)
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        terminal = self.__is_terminal()
        self.lives = 0 if terminal else 1
        observation = self.observation_pool.get_pooled_observations()
        self.global_step += 1
        return observation, reward, terminal

    def __is_terminal(self):
        return self.tetris.gameover

    def __is_over(self):
        return self.tetris.gameover

    def get_noop(self):
        return [1.0, 0.0]
