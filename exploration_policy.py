import numpy as np
import tensorflow as tf
import logging

class ExplorationPolicy:

    def __init__(self, args):
        self.egreedy_policy = args.egreedy
        self.initial_epsilon = args.epsilon
        self.epsilon = args.epsilon
        self.softmax_temp = args.softmax_temp
        self.use_dropout = args.use_dropout
        self.keep_percentage = args.keep_percentage
        self.global_step = 0
        self.annealed = args.annealed
        self.annealing_steps = 80000000
        self.pwyx_net = args.pwyx_net
        self.play_in_colours = args.play_in_colours
        self.oxygen_greedy = args.oxygen_greedy
        self.proba_oxygen = args.proba_oxygen
        self.nb_up_actions = args.nb_up_actions
        self.compteur_up_actions = 0

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

        if self.oxygen_greedy :
            action_indices = self.oxygen_greedy_choose(network_output_pi)
        elif self.egreedy_policy :
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

    def oxygen_greedy_choose(self, probs):
        if self.compteur_up_actions == 0 and np.random.rand(1)[0] > self.proba_oxygen :
            return self.multinomial_choose(probs)
        else :
            action_indexes = [2]*len(probs)
            self.compteur_up_actions = (self.compteur_up_actions + 1) % self.nb_up_actions
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
