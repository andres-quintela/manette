from multiprocessing import Process
from exploration_policy import Action


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
            ## pour FiGAR , changer le traitement des actions ici !s
            for i, (emulator, action_list) in enumerate(zip(self.emulators, self.variables[-1])):
                action = Action(i, action_list)
                new_s, reward, episode_over = emulator.next(action.current_action)
                if episode_over:
                    self.variables[0][i] = emulator.get_initial_state()
                else:
                    self.variables[0][i] = new_s
                self.variables[1][i] = reward
                self.variables[2][i] = episode_over
                while action.is_repeated() and not episode_over :
                    new_s, reward, episode_over = emulator.next(action.repeat())
                    if episode_over:
                        self.variables[0][i] = emulator.get_initial_state()
                    else:
                        self.variables[0][i] = new_s
                    self.variables[1][i] += reward
                    self.variables[2][i] = episode_over
                action.reset()
            count += 1
            self.barrier.put(True)
