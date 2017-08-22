import numpy as np

class ExplorationPolicy:

    def __init__(self, policies_dict = {"e_greedy":0.1}):
        self.policies_dict = policies_dict
        self.policies_list = policies_dict.keys()

    def choose_next_actions(self):
        pass

    def e_greedy_choose(self):
        pass

    def compute_entropy_loss(self):
        pass

    def dropout_forward(self):
        pass

    def noisy_forward(self):
        pass

    def default_forward(self):
        pass

    def boltzmann_choose(self):
        pass

    def multinomial_choose(self):
