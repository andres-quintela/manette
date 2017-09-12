import numpy as np
import tensorflow as tf
import logging

class Action :
    def __init__(self, i, l=[]):
        self.id = i
        self.repeated = False
        self.current_action = 0
        self.nb_repetitions_left = 0
        if l != [] : self.init_from_list(l)

    def init_from_list(self, l):
        self.current_action = np.argmax(l)
        self.nb_repetitions_left = max(l)
        if self.nb_repetitions_left > 0 :
            self.repeated = True


    def repeat(self):
        self.nb_repetitions_left -= 1
        if self.nb_repetitions_left == 0 :
            self.repeated = False
            self.current_action = 0
        return self.current_action

    def reset(self):
        self.repeated = False
        self.current_action = 0
        self.nb_repetitions_left = 0

    def is_repeated(self):
        return self.repeated

    @staticmethod
    def make_array(actions_list, nb_env, num_actions):
        actions_array = np.zeros((nb_env, num_actions), dtype = np.float32)
        for action in actions_list :
            actions_array[action.id][action.current_action] = action.nb_repetitions_left
        return actions_array

class ExplorationPolicy:

    def __init__(self, args):
        self.egreedy_policy = args.egreedy
        self.initial_epsilon = args.epsilon
        self.epsilon = args.epsilon
        self.softmax_temp = args.softmax_temp
        self.keep_percentage = args.keep_percentage
        self.global_step = 0
        self.annealed = args.annealed
        self.annealing_steps = 80000000
        self.oxygen_greedy = args.oxygen_greedy
        self.proba_oxygen = args.proba_oxygen
        self.nb_up_actions = args.nb_up_actions
        self.compteur_up_actions = 0
        self.FiGAR = args.FiGAR
        self.max_repetition = args.max_repetition
        self.nb_env = args.emulator_counts
        self.next_actions = np.array([Action(e) for e in range(self.nb_env)])

    def get_epsilon(self):
        if self.global_step <= self.annealing_steps:
            return self.initial_epsilon - (self.global_step * self.initial_epsilon / self.annealing_steps)
        else:
            return 0.0

    def choose_next_actions(self, network, num_actions, states, session):
        network_output_v, network_output_pi, network_output_rep = session.run(
            [network.output_layer_v,
             network.output_layer_pi, network.output_layer_rep],
            feed_dict={network.input_ph: states})

        if self.oxygen_greedy :
            action_indices = self.oxygen_greedy_choose(network_output_pi)
        elif self.egreedy_policy :
            action_indices = self.e_greedy_choose(network_output_pi)
        else :
            action_indices = self.multinomial_choose(network_output_pi)

        repetition_indices = self.choose_repetition(network_output_rep)

        for e in range(self.nb_env):
            self.next_actions[e].current_action = action_indices[e]
            self.next_actions[e].nb_repetitions_left = repetition_indices[e]
            if self.next_actions[e].nb_repetitions_left > 0 :
                self.next_actions[e].repeated = True

        self.global_step += len(network_output_pi)
        if self.annealed : self.epsilon = get_epsilon()

        return self.next_actions, network_output_v, network_output_pi

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
        probs = probs - np.finfo(np.float32).epsneg
        if self.compteur_up_actions == 0 and np.random.rand(1)[0] > self.proba_oxygen :
            return self.multinomial_choose(probs)
        else :
            action_indexes = [2 for i in range(10)] #exploration pour les 10 premmiers environments
            action_indexes += [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs[10:]]
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

    def choose_repetition(self, probs):
        return self.multinomial_choose(probs)
