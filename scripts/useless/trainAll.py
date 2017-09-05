import argparse
import os
import datetime
import json
import subprocess


def bool_arg(string):
    value = string.lower()
    if value == 'true':
        return True
    elif value == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError("Expected True or False, but got {}".format(string))

def save_args(args, folder, file_name='parameters.json'):
    args = vars(args)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, file_name), 'w') as f:
        return json.dump(args, f)

def create_cmd(args, game_name, path):
    cmd = ("python3 train.py -g "+game_name+" -df "+path+"/"+game_name+
                " -explo "+args.explo_policy+
                " -v "+str(args.visualize)+
                " --e "+str(args.e)+
                " --alpha "+str(args.alpha)+
                " -lr "+str(args.initial_lr)+
                " -lra "+str(args.lr_annealing_steps)+
                " --entropy "+str(args.entropy_regularisation_strength)+
                " --clip_norm "+str(args.clip_norm)+
                " --clip_norm_type "+str(args.clip_norm_type)+
                " --gamma "+str(args.gamma)+
                " --max_global_steps "+str(args.max_global_steps)+
                " --max_local_steps "+str(args.max_local_steps)+
                " --arch "+str(args.arch)+
                " --single_life_episodes "+str(args.single_life_episodes)+
                " -ec "+str(args.emulator_counts)+
                " -ew "+str(args.emulator_workers)+
                " -rs "+str(args.random_start))
    return cmd

def main(args):
    i = datetime.datetime.now()
    path = "logs/"+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+args.name_exp

    if not os.path.exists(path):
        os.makedirs(path)
    save_args(args, path)

    subprocess.call("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64", shell = True)
    subprocess.call(("export CUDA_VISIBLE_DEVICES="+str(args.device)), shell = True)

    if('po' in args.games):
        subprocess.call(create_cmd(args, "pong", path), shell = True)
    if('br' in args.games):
        subprocess.call(create_cmd(args, "breakout", path), shell = True)
    if('ms' in args.games):
        subprocess.call(create_cmd(args, "ms_pacman", path), shell = True)
    if('sp' in args.games):
        subprocess.call(create_cmd(args, "space_invaders", path), shell = True)
    if('mo' in args.games):
        subprocess.call(create_cmd(args, "montezuma_revenge", path), shell = True)
    if('se' in args.games):
        subprocess.call(create_cmd(args, "seaquest", path), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default= '', type=str, help='Name of the experiment', dest='name_exp')
    parser.add_argument('-explo', default='multi_0_1', type=str, help="Exploration policy to be used", dest="explo_policy")
    parser.add_argument('-g', default= 'po', type=str, help='Name of the games to train', dest='games')
    parser.add_argument('-d', '--device', default=2, type=int, help="Device to be used : 0,1 or 2", dest="device")
    parser.add_argument('-v', '--visualize', default=False, type=bool_arg, help="0: no visualization of emulator; 1: all emulators, for all actors, are visualized; 2: only 1 emulator (for one of the actors) is visualized", dest="visualize")
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
    parser.add_argument('--arch', default='NIPS', help="Which network architecture to use: from the NIPS or NATURE paper", dest="arch")
    parser.add_argument('--single_life_episodes', default=False, type=bool_arg, help="If True, training episodes will be terminated when a life is lost (for games)", dest="single_life_episodes")
    parser.add_argument('-ec', '--emulator_counts', default=32, type=int, help="The amount of emulators per agent. Default is 32.", dest="emulator_counts")
    parser.add_argument('-ew', '--emulator_workers', default=8, type=int, help="The amount of emulator workers per agent. Default is 8.", dest="emulator_workers")
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-rs', '--random_start', default=True, type=bool_arg, help="Whether or not to start with 30 noops for each env. Default True", dest="random_start")
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
