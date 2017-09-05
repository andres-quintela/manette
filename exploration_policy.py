import numpy as np
import tensorflow as tf

class ExplorationPolicy:

    def __init__(self, egreedy, epsilon, softmax_temp,
                 use_dropout, keep_percentage, annealed):
        self.egreedy_policy = egreedy
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.softmax_temp = softmax_temp
        self.use_dropout = use_dropout
        self.keep_percentage = keep_percentage
        self.global_step = 0
        self.annealed = annealed
        self.annealing_steps = 80000000

    def get_epsilon(self):
        if self.global_step <= self.annealing_steps:
            return self.initial_epsilon - (self.global_step * self.initial_epsilon / self.annealing_steps)
        else:
            return 0.0

    def choose_next_actions(self, network, num_actions, states, session):
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v,
             network.output_layer_pi],
            feed_dict={network.input_ph: states})

        if self.egreedy_policy :
            action_indices = self.e_greedy_choose(network_output_pi)
        else :
            action_indices = self.multinomial_choose(network_output_pi)
        new_actions = np.eye(num_actions)[action_indices]

        self.global_step += len(network_output_pi)
        if self.annealed : self.epsilon = get_epsilon()

        return new_actions, network_output_v, network_output_pi

    def e_greedy_choose(self, probs):
        """Sample an action from an action probability distribution output by
        the policy network using a greedy policy"""
        action_indexes = []
        for p in probs :
            if np.random.rand(1)[0] < self.epsilon :
                i = np.random.randint(0,len(p))
                action_indexes.append(i)
            else :
                action_indexes.append(np.argmax(p))
        return action_indexes

    def multinomial_choose(self, probs):
        """Sample an action from an action probability distribution output by
        the policy network using a multinomial law."""
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

    def compute_entropy_term(self):
       pass
