from networks import *
class PolicyVNetwork(Network):

    def __init__(self, conf):
        """ Set up remaining layers, objective and loss functions, gradient
        compute and apply ops, network parameter synchronization ops, and
        summary ops. """

        super(PolicyVNetwork, self).__init__(conf)

        self.entropy_regularisation_strength = conf['entropy_regularisation_strength']
        self.softmax_temp = conf['softmax_temp']
        self.op = Operations(conf)
        self.total_repetitions = conf['nb_choices']

        with tf.device(conf['device']):
            with tf.name_scope('Training'):

                with tf.name_scope('Critic'):
                    self.critic_target_ph = tf.placeholder("float32", [None], name='target')
                    # Final critic layer
                    _, _, self.output_layer_v = self.op.fc('critic_output', self.output, 1, activation="linear")
                    self.output_layer_v = tf.reshape(self.output_layer_v, [-1])
                    # Advantage critic
                    self.critic_loss = tf.subtract(self.critic_target_ph, self.output_layer_v)
                    self.critic_loss_mean = tf.reduce_mean(tf.scalar_mul(0.25, tf.pow(self.critic_loss, 2)), name='mean_critic_loss')


                with tf.name_scope('Actor'):
                    # Final actor layer
                    _, _, self.output_layer_pi = self.op.softmax('actor_output', self.output, self.num_actions, self.softmax_temp)
                    # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                    self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                      name='actor_output_log_policy')
                    # Entropy: sum_a (-p_a ln p_a)
                    self.output_layer_entropy_actor = tf.reduce_sum(tf.multiply(
                            tf.constant(-1.0),
                            tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)
                    self.log_output_selected_action = tf.reduce_sum(
                            tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                            reduction_indices=1)
                    self.actor_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength, self.output_layer_entropy_actor)


                with tf.name_scope('Repetition'):
                    # Final repetition layer
                    _, _, self.output_layer_rep = self.op.softmax('repetition_output', self.output, self.total_repetitions, self.softmax_temp)
                    self.log_output_layer_rep = tf.log(tf.add(self.output_layer_rep, tf.constant(1e-30)),
                                                    name='repetition_output_log_policy')
                    self.output_layer_entropy_rep = tf.reduce_sum(tf.multiply(
                            tf.constant(-1.0),
                            tf.multiply(self.output_layer_rep, self.log_output_layer_rep)), reduction_indices=1, name='entropy_rep')
                    self.log_output_selected_repetition = tf.reduce_sum(
                            tf.multiply(self.log_output_layer_rep, self.selected_repetition_ph),
                            reduction_indices=1)
                    self.log_repetition_mean = tf.reduce_mean(self.log_output_selected_repetition)
                    self.rep_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength, self.output_layer_entropy_rep, name='rep_entropy_term')

                with tf.name_scope('ComputeLoss'):
                    self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')
                    self.actor_objective_advantage_term = tf.multiply(
                                        tf.add(self.log_output_selected_action, self.log_output_selected_repetition),
                                        self.adv_actor_ph, name='advantage')
                    self.actor_advantage_mean = tf.reduce_mean(self.actor_objective_advantage_term)

                    self.entropy_term = tf.add(self.actor_objective_entropy_term, self.rep_objective_entropy_term, name='entropy_term')

                    self.actor_objective_mean = tf.reduce_mean(tf.multiply(tf.constant(-1.0),
                                                tf.add(self.actor_objective_advantage_term, self.entropy_term)),
                                                name='mean_actor_objective')

                    # Loss scaling is used because the learning rate was initially runed tuned to be used with
                    # max_local_steps = 5 and summing over timesteps, which is now replaced with the mean.
                    self.loss = tf.scalar_mul(self.loss_scaling, self.actor_objective_mean + self.critic_loss_mean)


class NIPSPolicyVNetwork(PolicyVNetwork, NIPSNetwork):
    pass

class BayesianPolicyVNetwork(PolicyVNetwork, BayesianNetwork):
    pass

class PpwwyyxxPolicyVNetwork(PolicyVNetwork, PpwwyyxxNetwork):
    pass

class NaturePolicyVNetwork(PolicyVNetwork, NatureNetwork):
    pass
