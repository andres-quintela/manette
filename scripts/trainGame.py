import argparse
import os
import datetime
import json
import subprocess

def create_cmd(args, game, path, v):
    cmd = ("python3 train.py -g "+game+" -df "+path+"/"+v+
                " -"+args.param+" "+v+
                " --max_global_steps "+str(args.max_global_steps)+
                " -lra "+str(args.lr_annealing_steps))
    return cmd

def parse_games(args):
    game = ""
    if('po' in args.game_name): game = "pong"
    if('br' in args.game_name): game = "breakout"
    if('ms' in args.game_name): game = "ms_pacman"
    if('sp' in args.game_name): game = "space_invaders"
    if('mo' in args.game_name): game = "montezuma_revenge"
    if('se' in args.game_name): game = "seaquest"
    return game

def main(args):
    i = datetime.datetime.now()
    path = "logs/"+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+args.game_name+args.name_exp
    if not os.path.exists(path):
        os.makedirs(path)
    game = parse_games(args)

    values_list = [v for v in args.values.split("/")]
    for v in values_list :
        subprocess.call(create_cmd(args, game, path, v), shell = True)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default= '', type=str, help='Name of the experiment', dest='name_exp')
    parser.add_argument('-g', default= 'po', type=str, help='Name of the game to train', dest='game_name')
    parser.add_argument('-p', default= 'lr', type=str, help='Name of the parameter to change', dest='param')
    parser.add_argument('-val', default='1', type=str, help='Values of the parameter', dest='values')
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
