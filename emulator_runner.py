from multiprocessing import Process
from exploration_policy import Action
import logging


class EmulatorRunner(Process):

    def __init__(self, id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulators = emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (emulator, action, rep) in enumerate(zip(self.emulators, self.variables[-2], self.variables[-1])):
                macro_action = Action(i, action, rep)
                logging.info("current_action : "+str(macro_action.current_action))
                new_s, reward, episode_over = emulator.next(macro_action.current_action)
                if episode_over:
                    self.variables[0][i] = emulator.get_initial_state()
                else:
                    self.variables[0][i] = new_s
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
            count += 1
            self.barrier.put(True)
