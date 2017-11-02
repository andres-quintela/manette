from multiprocessing import Process
from exploration_policy import Action
from scipy.misc import imsave
import numpy as np

class EmulatorRunner(Process):

    def __init__(self, i, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = i
        self.emulators = emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (emulator, action, rep) in enumerate(zip(self.emulators, self.variables[-2], self.variables[-1])):
                macro_action = Action(i, action, rep)
                new_s, reward, episode_over = emulator.next(macro_action.current_action)
                if episode_over:
                    self.variables[0][i] = emulator.get_initial_state()
                else:
                    self.variables[0][i] = new_s
                    # z = np.zeros((84, 84, 3), dtype=np.uint8)
                    # z[:,:,0], z[:,:,1], z[:,:,2] = new_s[:, :, 0], new_s[:, :, 4], new_s[:, :, 8]
                    # imsave('ns1.jpg', z)
                    # z[:,:,0], z[:,:,1], z[:,:,2] = new_s[:, :, 1], new_s[:, :, 5], new_s[:, :, 9]
                    # imsave('ns2.jpg', z)
                    # z[:,:,0], z[:,:,1], z[:,:,2] = new_s[:, :, 2], new_s[:, :, 6], new_s[:, :, 10]
                    # imsave('ns3.jpg', z)
                    # z[:,:,0], z[:,:,1], z[:,:,2] = new_s[:, :, 3], new_s[:, :, 7], new_s[:, :, 11]
                    # imsave('ns4.jpg', z)
                    # z = np.zeros((84, 84), dtype=np.uint8)
                    # z = new_s[ :, :, 0]
                    # imsave('runner1.jpg', z)
                    # z = new_s[ :, :, 1]
                    # imsave('runner2.jpg', z)
                    # z = new_s[ :, :, 2]
                    # imsave('runner3.jpg', z)
                    # z = new_s[ :, :, 3]
                    # imsave('runner4.jpg', z)
                self.variables[1][i] = reward
                self.variables[2][i] = episode_over
                while macro_action.is_repeated() and not episode_over :
                    new_s, reward, episode_over = emulator.next(macro_action.repeat())
                    if episode_over:
                        self.variables[0][i] = emulator.get_initial_state()
                    else:
                        self.variables[0][i] = new_s
                    self.variables[1][i] += reward
                    self.variables[2][i] = episode_over
                macro_action.reset()
            self.barrier.put(True)
