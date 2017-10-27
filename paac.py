import time
from multiprocessing import Queue
from multiprocessing.sharedctypes import RawArray
from ctypes import c_uint, c_float
from actor_learner import *
import logging
from logger_utils import variable_summaries
import numpy as np

from emulator_runner import EmulatorRunner
from exploration_policy import Action
from runners import Runners


class PAACLearner(ActorLearner):
    def __init__(self, network_creator, environment_creator, explo_policy, args):
        super(PAACLearner, self).__init__(network_creator, environment_creator, explo_policy, args)
        self.workers = args.emulator_workers
        self.total_repetitions = args.max_repetition

        #add the parameters to tensorboard
        sess = tf.InteractiveSession()
        file_args = open(args.debugging_folder+"args.json", 'r')
        text = str(file_args.read())
        summary_op = tf.summary.text('text', tf.convert_to_tensor(text))
        text = sess.run(summary_op)
        self.summary_writer.add_summary(text,0)
        self.summary_writer.flush()
        sess.close()


    def _get_shared(self, array, dtype=c_float):
        """
        Returns a RawArray backed numpy array that can be shared between processes.
        :param array: the array to be shared
        :param dtype: the RawArray dtype to use
        :return: the RawArray backed numpy array """

        shape = array.shape
        shared = RawArray(dtype, array.reshape(-1))
        return np.frombuffer(shared, dtype).reshape(shape)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values"""

        counts, bin_edges = np.histogram(values, bins=bins)
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        bin_edges = bin_edges[1:]

        for edge in bin_edges : hist.bucket_limit.append(edge)
        for c in counts : hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def log_values(self, values, tag, length = 50, timestep = 500):
        if len(values) > length and self.global_step % timestep == 0 :
            mean = np.mean(values[-50:])
            std = np.std(values[-50:])
            summary = tf.Summary(value=[
                tf.Summary.Value(tag=tag+'/mean', simple_value=mean),
                tf.Summary.Value(tag=tag+'/min', simple_value=min(values[-50:])),
                tf.Summary.Value(tag=tag+'/max', simple_value=max(values[-50:])),
                tf.Summary.Value(tag=tag+'/std', simple_value=std),
                tf.Summary.Value(tag=tag+'/std_over_mean', simple_value=min(2, np.absolute(std/mean)))
            ])
            self.summary_writer.add_summary(summary, self.global_step)
            self.summary_writer.flush()


    def train(self):
        """ Main actor learner loop for parallel advantage actor critic learning."""

        self.global_step = self.init_network()
        global_step_start = self.global_step
        counter = 0
        total_rewards = []
        total_steps = []

        logging.debug("Starting training at Step {}".format(self.global_step))

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
            print('step : '+str(self.global_step))

            loop_start_time = time.time()

            total_action_rep = np.zeros((self.num_actions, self.total_repetitions))

            nb_actions = 0

            max_local_steps = self.max_local_steps
            for t in range(max_local_steps):

                new_actions, new_repetitions, readouts_v_t, readouts_pi_t = self.explo_policy.choose_next_actions(self.network, self.num_actions, shared_states, self.session)

                actions_sum += new_actions

                for e in range(self.emulator_counts) :
                    nb_actions += np.argmax(new_repetitions[e]) + 1

                # sharing the actions and repetitions to the different threads
                for z in range(new_actions.shape[0]): shared_actions[z] = new_actions[z]
                for z in range(new_repetitions.shape[0]): shared_rep[z] = new_repetitions[z]

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

                    #rempli le tableau pour l'histogramme des actions - repetitions
                    a = np.argmax(new_actions[e])
                    r = np.argmax(new_repetitions[e])
                    total_action_rep[a][r] += 1

                    if episode_over:
                        total_rewards.append(total_episode_rewards[e])
                        total_steps.append(emulator_steps[e])
                        episode_summary = tf.Summary(value=[
                            tf.Summary.Value(tag='rl/reward', simple_value=total_episode_rewards[e]),
                            tf.Summary.Value(tag='rl/episode_length', simple_value=emulator_steps[e])
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
            self.summary_writer.flush()

            self.log_values(total_rewards, 'rewards_per_episode')
            self.log_values(total_steps, 'steps_per_episode')

            #ajout de l'histogramme des actions
            nb_a = [ sum(a) for a in total_action_rep]
            total_action_rep_trans = np.transpose(total_action_rep)
            nb_r = [ sum(r) for r in total_action_rep_trans ]
            histo_a, histo_r = [], []
            for i in range(self.num_actions) : histo_a += [i]*int(nb_a[i])
            for i in range(self.total_repetitions) : histo_r += [i]*int(nb_r[i])

            self.log_histogram('actions', np.array(histo_a), self.global_step)
            self.log_histogram('repetitions', np.array(histo_r), self.global_step)

            counter += 1
            if counter % (2048 / self.emulator_counts) == 0:
                curr_time = time.time()
                last_ten = 0.0 if len(total_rewards) < 1 else np.mean(total_rewards[-10:])
                steps_per_sec = self.max_local_steps * self.emulator_counts / (curr_time - loop_start_time)
                actions_per_s = nb_actions / (curr_time - loop_start_time)
                average_steps_per_sec = (self.global_step - global_step_start) / (curr_time - start_time)
                logging.info("Ran {} steps, at {} steps/s ({} steps/s avg), last 10 rewards avg {}"
                             .format(self.global_step, steps_per_sec, average_steps_per_sec, last_ten))

                stats_summary = tf.Summary(value=[
                    tf.Summary.Value(tag='stats/steps_per_s', simple_value=steps_per_sec),
                    tf.Summary.Value(tag='stats/average_steps_per_s', simple_value=average_steps_per_sec),
                    tf.Summary.Value(tag='stats/actions_per_s', simple_value=actions_per_s)
                ])
                self.summary_writer.add_summary(stats_summary, self.global_step)
                self.summary_writer.flush()

            self.save_vars()

        self.cleanup()

    def cleanup(self):
        super(PAACLearner, self).cleanup()
        self.runners.stop()
