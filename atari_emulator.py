import numpy as np
from ale_python_interface import ALEInterface
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


class AtariEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.ale = ALEInterface()
        self.ale.setInt(b"random_seed", args.random_seed * (actor_id +1))
        # For fuller control on explicit action repeat (>= ALE 0.5.0)
        self.ale.setFloat(b"repeat_action_probability", 0.0)
        # Disable frame_skip and color_averaging
        # See: http://is.gd/tYzVpj
        self.ale.setInt(b"frame_skip", 1)
        self.ale.setBool(b"color_averaging", False)
        full_rom_path = args.rom_path + "/" + args.game + ".bin"
        self.ale.loadROM(str.encode(full_rom_path))
        self.legal_actions = self.ale.getMinimalActionSet()
        self.screen_width, self.screen_height = self.ale.getScreenDims()
        self.lives = self.ale.lives()

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
        self.ale.getScreenGrayscale(self.gray_screen)
        if self.rgb :
            self.ale.getScreenRGB(self.rgb_screen)
        if self.call_on_new_frame:
            self.ale.getScreenRGB(self.rgb_screen)
            self.on_new_frame(self.rgb_screen)
        if self.rgb :
            return self.rgb_screen
        return self.gray_screen

    def on_new_frame(self, frame):
        pass

    def __new_game(self):
        """ Restart game """
        self.ale.reset_game()
        self.lives = self.ale.lives()
        if self.random_start:
            wait = random.randint(0, MAX_START_WAIT)
            for _ in range(wait):
                self.ale.act(self.legal_actions[0])

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
            reward += self.ale.act(self.legal_actions[a])
        # Only need to add the last FRAMES_IN_POOL frames to the frame pool
        for i in range(FRAMES_IN_POOL):
            reward += self.ale.act(self.legal_actions[a])
            img = self.__get_screen_image()
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
        cv2.imwrite('dataset/x/'+str(self.compteur)+'.jpg', self.frame_pool.frame_pool[0])
        with open('dataset/y.txt', 'a') as f :
            f.write('x/'+str(self.compteur+1)+'.jpg : '+str(action)+'\n')
        self.compteur+=1
        reward = self.__action_repeat(action)
        self.observation_pool.new_observation(self.frame_pool.get_processed_frame())
        terminal = self.__is_terminal()
        self.lives = self.ale.lives()
        observation = self.observation_pool.get_pooled_observations()
        self.global_step += 1
        return observation, reward, terminal

    def __is_terminal(self):
        if self.single_life_episodes:
            return self.__is_over() or (self.lives > self.ale.lives())
        else:
            return self.__is_over()

    def __is_over(self):
        return self.ale.game_over()

    def get_noop(self):
        return [1.0, 0.0]
