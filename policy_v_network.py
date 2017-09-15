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
        self.max_repetition = conf['max_repetition']
        self.total_repetitions = self.max_repetition + 1

        with tf.device(conf['device']):
            with tf.name_scope(self.name):

                self.critic_target_ph = tf.placeholder(
                    "float32", [None], name='target')
                self.adv_actor_ph = tf.placeholder("float", [None], name='advantage')
                #self.adv_rep_ph = tf.placeholder("float", [None], name='advantage_rep')

                # Final actor layer
                layer_name = 'actor_output'
                _, _, self.output_layer_pi = self.op.softmax(layer_name, self.output, self.num_actions, self.softmax_temp)
                print("OUTPUT LAYER PI")
                print(self.output_layer_pi)
                # Final repetition layer
                _, _, self.output_layer_rep = self.op.softmax('repetition_output', self.output, self.total_repetitions, self.softmax_temp)
                print("OUTPUT LAYER REP")
                print(self.output_layer_rep)
                # Final critic layer
                _, _, self.output_layer_v = self.op.fc('critic_output', self.output, 1, activation="linear")

                # Avoiding log(0) by adding a very small quantity (1e-30) to output.
                self.log_output_layer_pi = tf.log(tf.add(self.output_layer_pi, tf.constant(1e-30)),
                                                  name=layer_name + '_log_policy')
                print("LOG OUTPUT LAYER PI")
                print(self.log_output_layer_pi)
                self.log_output_layer_rep = tf.log(tf.add(self.output_layer_rep, tf.constant(1e-30)),
                                                name='repetition_output_log_policy')
                print("LOG OUTPUT LAYER REP")
                print(self.log_output_layer_rep)


                # Entropy: sum_a (-p_a ln p_a)
                self.output_layer_entropy = tf.reduce_sum(tf.multiply(
                    tf.constant(-1.0),
                    tf.multiply(self.output_layer_pi, self.log_output_layer_pi)), reduction_indices=1)

                self.output_layer_v = tf.reshape(self.output_layer_v, [-1])

                # Advantage critic
                self.critic_loss = tf.subtract(self.critic_target_ph, self.output_layer_v) ## a changer

                self.log_output_selected_action = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_pi, self.selected_action_ph),
                    reduction_indices=1)
                print("LOG OUTPUT SELECTED ACTION")
                print(self.log_output_selected_action)

                self.log_output_selected_repetition = tf.reduce_sum(
                    tf.multiply(self.log_output_layer_rep, self.selected_repetition_ph),
                    reduction_indices=1)
                print("LOG OUTPUT SELECTED REPETITION")
                print(self.log_output_selected_repetition)

                self.log_repetition_mean = tf.reduce_mean(self.log_output_selected_repetition)

                self.actor_objective_advantage_term = tf.multiply(
                                    tf.add(self.log_output_selected_action, self.log_output_selected_repetition),
                                    self.adv_actor_ph)

                self.actor_advantage_mean = tf.reduce_mean(self.actor_objective_advantage_term)

                self.actor_objective_entropy_term = tf.multiply(self.entropy_regularisation_strength, self.output_layer_entropy)



                self.actor_objective_mean = tf.reduce_mean(tf.multiply(tf.constant(-1.0),
                                                                      tf.add(self.actor_objective_advantage_term, self.actor_objective_entropy_term)),
                                                           name='mean_actor_objective')
                print("ACTOR OBJECTIVE MEAN")
                print(self.actor_objective_mean)

                self.critic_loss_mean = tf.reduce_mean(tf.scalar_mul(0.25, tf.pow(self.critic_loss, 2)), name='mean_critic_loss')

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
