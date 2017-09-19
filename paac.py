import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging
from logger_utils import variable_summaries

from emulator_runner import EmulatorRunner
from exploration_policy import Action
from runners import Runners
import numpy as np
import sys

class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, explo_policy, args):
        logging.info('#######Init PAAC')
        super(PAACLearner, self).__init__(network_creator, environment_creator, explo_policy, args)
        self.workers = args.emulator_workers
        self.max_repetition = args.max_repetition
        self.total_repetitions = self.max_repetition + 1


    @staticmethod
    def choose_next_actions(network, num_actions, states, session):
        network_output_v, network_output_pi = session.run(
            [network.output_layer_v,
             network.output_layer_pi],
            feed_dict={network.input_ph: states})

        action_indices = PAACLearner.__sample_policy_action(network_output_pi)

        new_actions = np.eye(num_actions)[action_indices]

        return new_actions, network_output_v, network_output_pi

    def __choose_next_actions(self, states):
        return PAACLearner.choose_next_actions(self.network, self.num_actions, states, self.session)

    @staticmethod
    def __sample_policy_action(probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        action_indexes = [int(np.nonzero(np.random.multinomial(1, p))[0]) for p in probs]
        return action_indexes

    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array
        """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        print('paac shared array length=',len(shared))
        return np.frombuffer(shared, dtype).reshape(shape)

    def train(self):
        """
        Main actor learner loop for parallel advantage actor critic learning.
        """

        self.global_step = self.init_network()

        logging.debug("Starting training at Step {}".format(self.global_step))
        counter = 0

        global_step_start = self.global_step

        total_rewards = []
        total_steps = []


        # state, reward, episode_over, action, repetition
        variables = [(np.asarray([emulator.get_initial_state() for emulator in self.emulators], dtype=np.uint8)),
                     (np.zeros(self.emulator_counts, dtype=np.float32)),
                     (np.asarray([False] * self.emulator_counts, dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.num_actions), dtype=np.float32)),
                     (np.zeros((self.emulator_counts, self.total_repetitions), dtype=np.float32))]

        self.runners = Runners(EmulatorRunner, self.emulators, self.workers, variables)
        self.runners.start()
        shared_states, shared_rewards, shared_episode_over, shared_actions, shared_rep = self.runners.get_shared_variables()

        summaries_op = tf.summary.merge_all()

        emulator_steps = [0] * self.emulator_counts
        total_episode_rewards = self.emulator_counts * [0]

        actions_sum = np.zeros((self.emulator_counts, self.num_actions))
        y_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        adv_batch = np.zeros((self.max_local_steps, self.emulator_counts))
        rewards = np.zeros((self.max_local_steps, self.emulator_counts))
        states = np.zeros([self.max_local_steps] + list(shared_states.shape), dtype=np.uint8)
        actions = np.zeros((self.max_local_steps, self.emulator_counts, self.num_actions))
        repetitions = np.zeros((self.max_local_steps, self.emulator_counts, self.total_repetitions))
        values = np.zeros((self.max_local_steps, self.emulator_counts))
        episodes_over_masks = np.zeros((self.max_local_steps, self.emulator_counts))

        start_time = time.time()

        while self.global_step < self.max_global_steps:

            loop_start_time = time.time()

            total_action_rep = np.zeros((self.num_actions, self.total_repetitions))

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):
                #next_actions, readouts_v_t, readouts_pi_t = self.__choose_next_actions(shared_states)
                new_actions, new_repetitions, readouts_v_t, readouts_pi_t = self.explo_policy.choose_next_actions(self.network, self.num_actions, shared_states, self.session)

                actions_sum += new_actions

                for z in range(new_actions.shape[0]):
                    shared_actions[z] = new_actions[z]
                for z in range(new_repetitions.shape[0]):
                    shared_rep[z] = new_repetitions[z]

                shared_rep = new_repetitions
                #logging.info("SHARED ACTIONS : "+str(shared_actions))
                #logging.info("SHARED REP : "+str(shared_rep))

                actions[t] = new_actions
                values[t] = readouts_v_t
                states[t] = shared_states
                repetitions[t] = new_repetitions

                # Start updating all environments with next_actions
                self.runners.update_environments()
                self.runners.wait_updated()
                # Done updating all environments, have new states, rewards and is_over

                episodes_over_masks[t] = 1.0 - shared_episode_over.astype(np.float32)

                for e, (actual_reward, episode_over) in enumerate(zip(shared_rewards, shared_episode_over)):
                    total_episode_rewards[e] += actual_reward
                    actual_reward = self.rescale_reward(actual_reward)
                    rewards[t, e] = actual_reward

                    emulator_steps[e] += np.argmax(new_repetitions[e]) + 1
                    self.global_step += 1

                    #ici : rempli le tableau pour l'histogramme des actions - repetitions
                    a = np.argmax(new_actions[e])
                    r = np.argmax(new_repetitions[e])
                    total_action_rep[a][r] += 1

                    if episode_over:
                        total_rewards.append(total_episode_rewards[e])
                        total_steps.append(emulator_steps[e])
                        episode_summary = tf.Summary(value=[
                            tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e]),
                            tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e])
                            #tf.Summary.Value(tag='rl/loss', simple_value=self.network.loss)
                        ])
                        self.summary_writer.add_summary(episode_summary, self.global_step)
                        self.summary_writer.flush()
                        total_episode_rewards[e] = 0
                        emulator_steps[e] = 0
                        actions_sum[e] = np.zeros(self.num_actions)


            nest_state_value = self.session.run(
                self.network.output_layer_v,
                feed_dict={self.network.input_ph: shared_states})

            estimated_return = np.copy(nest_state_value)

            for t in reversed(range(max_local_steps)):
                estimated_return = rewards[t] + self.gamma * estimated_return * episodes_over_masks[t]
                ## ici changer des choses , puissance gamma
                y_batch[t] = np.copy(estimated_return)
                adv_batch[t] = estimated_return - values[t]

            flat_states = states.reshape([self.max_local_steps * self.emulator_counts] + list(shared_states.shape)[1:])
            flat_y_batch = y_batch.reshape(-1)
            flat_adv_batch = adv_batch.reshape(-1)
            flat_actions = actions.reshape(max_local_steps * self.emulator_counts, self.num_actions)
            flat_rep = repetitions.reshape(max_local_steps * self.emulator_counts, self.total_repetitions)

            lr = self.get_lr()
            feed_dict = {self.network.input_ph: flat_states,
                         self.network.critic_target_ph: flat_y_batch,
                         self.network.selected_action_ph: flat_actions,
                         self.network.selected_repetition_ph: flat_rep,
                         self.network.adv_actor_ph: flat_adv_batch,
                         self.learning_rate: lr}

            _, summaries = self.session.run(
                [self.train_step, summaries_op],
                feed_dict=feed_dict)

            self.summary_writer.add_summary(summaries, self.global_step)
            param_summary = tf.Summary(value=[
                tf.Summary.Value(tag='parameters/lr', simple_value=lr)
            ])
            self.summary_writer.add_summary(param_summary, self.global_step)

            if len(total_rewards) > 50 and self.global_step % 500 == 0 :
                mean = np.mean(total_rewards[-50:])
                std = np.std(total_rewards[-50:])
                rewards_summary = tf.Summary(value=[
                    tf.Summary.Value(tag='rewards_env/mean', simple_value=mean),
                    tf.Summary.Value(tag='rewards_env/min', simple_value=min(total_rewards[-50:])),
                    tf.Summary.Value(tag='rewards_env/max', simple_value=max(total_rewards[-50:])),
                    tf.Summary.Value(tag='rewards_env/std', simple_value=std),
                    tf.Summary.Value(tag='rewards_env/std_over_mean', simple_value=min(2, np.absolute(std/mean)))
                ])
                self.summary_writer.add_summary(rewards_summary, self.global_step)

            if len(total_steps) > 50 and self.global_step % 500 == 0 :
                mean_step = np.mean(total_steps[-50:])
                std_step = np.std(total_steps[-50:])
                steps_summary = tf.Summary(value=[
                    tf.Summary.Value(tag='steps_per_episode/mean', simple_value=mean_step),
                    tf.Summary.Value(tag='steps_per_episode/min', simple_value=min(total_steps[-50:])),
                    tf.Summary.Value(tag='steps_per_episode/max', simple_value=max(total_steps[-50:])),
                    tf.Summary.Value(tag='steps_per_episode/std', simple_value=std_step),
                    tf.Summary.Value(tag='steps_per_episode/std_over_mean', simple_value=min(2, np.absolute(std_step/mean_step)))
                ])
                self.summary_writer.add_summary(steps_summary, self.global_step)

            #histogramme des actions
            nb_a = [ sum(a) for a in total_action_rep]
            total_action_rep_trans = np.transpose(total_action_rep)
            nb_r = [ sum(r) for r in total_action_rep_trans ]
            histo_a = []
            for i in range(self.num_actions) :
                histo_a += [i]*int(nb_a[i])
            histo_r = []
            for i in range(self.total_repetitions) :
                histo_r += [i]*int(nb_r[i])
            
            histo_a_var = tf.Variable(histo_a)
            histo_r_var = tf.Variable(histo_r)

            tf.summary.histogram("actions", histo_a_var)
            tf.summary.histogram("repetitions", histo_r_var)

            self.summary_writer.flush()

            counter += 1

            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                global_steps = self.global_step
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(global_steps,
                                     self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time),
                                     (global_steps - global_step_start) / (curr_time - start_time),
                                     last_ten))
            self.save_vars()

        self.cleanup()

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()
