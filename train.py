import argparse
import logging
import sys
import signal
import os
import copy

import environment_creator
from exploration_policy import ExplorationPolicy
from paac import PAACLearner
from policy_v_network import NaturePolicyVNetwork, NIPSPolicyVNetwork, BayesianPolicyVNetwork, PpwwyyxxPolicyVNetwork
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def main(args):
    logging.debug('Configuration: {}'.format(args))

    explo_policy = ExplorationPolicy(args)
    print('Repetition table : '+str(explo_policy.tab_rep))

    network_creator, env_creator = get_network_and_environment_creator(args, explo_policy)

    learner = PAACLearner(network_creator, env_creator, explo_policy, args)

    setup_kill_signal_handler(learner)

    logging.info('Starting training')
    learner.train()
    logging.info('Finished training')


def setup_kill_signal_handler(learner):
    main_process_pid = os.getpid()

    def signal_handler(signal, frame):
        if os.getpid() == main_process_pid:
            logging.info('Signal ' + str(signal) + ' detected, cleaning up.')
            learner.cleanup()
            logging.info('Cleanup completed, shutting down...')
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def get_network_and_environment_creator(args, explo_policy, random_seed=3):
    env_creator = environment_creator.EnvironmentCreator(args)
    num_actions = env_creator.num_actions
    args.num_actions = num_actions
    args.random_seed = random_seed

    network_conf = {'num_actions': num_actions,
                    'entropy_regularisation_strength': args.entropy_regularisation_strength,
                    'device': args.device,
                    'clip_norm': args.clip_norm,
                    'clip_norm_type': args.clip_norm_type,
                    'softmax_temp' : explo_policy.softmax_temp,
                    'keep_percentage' : explo_policy.keep_percentage,
                    'rgb' : args.rgb,
                    'activation' : args.activation,
                    'alpha_leaky_relu' : args.alpha_leaky_relu,
                    'max_repetition' : args.max_repetition,
                    'nb_choices': args.nb_choices}

    if args.arch == 'PWYX' :
        network = PpwwyyxxPolicyVNetwork
    elif args.arch == 'BAYESIAN' :
        network = BayesianPolicyVNetwork
    elif args.arch == 'NIPS':
        network = NIPSPolicyVNetwork
    else:
        network = NaturePolicyVNetwork

    def network_creator(name='local_learning'):
        nonlocal network_conf
        copied_network_conf = copy.copy(network_conf)
        copied_network_conf['name'] = name
        return network(copied_network_conf)

    return network_creator, env_creator

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default='pong', help='Name of game', dest='game')
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    parser.add_argument('--rom_path', default='./atari_roms', help='Directory where the game roms are located (needed for ALE environment)', dest="rom_path")
    parser.add_argument('-v', '--visualize', default=0, type=int, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
    parser.add_argument('--e', default=0.1, type=float, help="Epsilon for the Rmsprop and Adam optimizers", dest="e")
    parser.add_argument('--alpha', default=0.99, type=float, help="Discount factor for the history/coming gradient, for the Rmsprop optimizer", dest="alpha")
    parser.add_argument('-lr', '--initial_lr', default=0.0224, type=float, help="Initial value for the learning rate. Default = 0.0224", dest="initial_lr")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    parser.add_argument('--entropy', default=0.02, type=float, help="Strength of the entropy regularization term (needed for actor-critic)", dest="entropy_regularisation_strength")
    parser.add_argument('--clip_norm', default=3.0, type=float, help="If clip_norm_type is local/global, grads will be clipped at the specified maximum (avaerage) L2-norm", dest="clip_norm")
    parser.add_argument('--clip_norm_type', default="global", help="Whether to clip grads by their norm or not. Values: ignore (no clipping), local (layer-wise norm), global (global norm)", dest="clip_norm_type")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor", dest="gamma")
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('--max_local_steps', default=5, type=int, help="Number of steps to gain experience from before every update.", dest="max_local_steps")
    parser.add_argument('--arch', default='NIPS', help="Which network architecture to use: from the NIPS or NATURE paper, or PWYX, or LSTM, or BAYESIAN ie dropout", dest="arch")
    parser.add_argument('--single_life_episodes', action='store_true', help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', action='store_true', help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")

    parser.add_argument('--egreedy', action='store_true', help="If True, e-greedy policy is used to  choose actions", dest="egreedy")
    parser.add_argument('--epsilon', default=0.05, type=float, help="Epsilon for the egreedy policy", dest='epsilon')
    parser.add_argument('--softmax_temp', default=1.0, type=float, help="Softmax temperature for the Boltzmann action policy", dest='softmax_temp' )
    parser.add_argument('--annealed', action='store_true', help="If True, the parameters for explo_policy are annealed toward zero", dest="annealed")
    parser.add_argument('--annealed_steps', default=80000000, type=int, help="Nb of global steps during which epsilon will be linearly annealed towards zero", dest="annealed_steps")
    parser.add_argument('--keep_percentage', default=0.9, type=float, help="keep percentage when dropout is used", dest='keep_percentage' )
    parser.add_argument('--rgb', action='store_true', help="If True, RGB images are given to the agent", dest="rgb")
    parser.add_argument('--max_repetition', default=0, type=int, help="Maximum number of repetition for FiGAR", dest="max_repetition")
    parser.add_argument('--nb_choices', default=1, type=int, help="Number of possible repetitions", dest="nb_choices")
    parser.add_argument('--checkpoint_interval', default=1000000, type=int, help="Interval of steps btw checkpoints", dest="checkpoint_interval")
    parser.add_argument('--activation', default='relu', type=str, help="activation function for the network", dest="activation")
    parser.add_argument('--alpha_leaky_relu', default=0.1, type=float, help="coef for leaky relu", dest="alpha_leaky_relu")

    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()

    import logger_utils
    logger_utils.save_args(args, args.debugging_folder)
    logging.debug(args)

    main(args)
