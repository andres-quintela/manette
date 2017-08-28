import argparse
import os
import datetime
import json
import subprocess

def create_cmd(args, game, path, v):
    cmd = ("python3 train.py -g "+game+" -df "+path+"/"+game+v+
                " -"+args.param+" "+v+
                " --max_global_steps "+str(args.max_global_steps)+
                " -lra "+str(args.lr_annealing_steps))
    return cmd

def create_chpt_cmd(args, game, path, v):
    cmd = ("nohup python3 scripts/checkpoints.py "+
                " -df "+path+"/"+game+v+"/"
                " -t "+str(args.time)+
                " &> nohupLogs/saveCheckpoints"+args.name_exp+".out &")
    return cmd

def parse_games(args):
    games = []
    if('po' in args.games): games.append("pong")
    if('br' in args.games): games.append("breakout")
    if('ms' in args.games): games.append("ms_pacman")
    if('sp' in args.games): games.append("space_invaders")
    if('mo' in args.games): games.append("montezuma_revenge")
    if('se' in args.games): games.append("seaquest")
    return games

def main(args):
    i = datetime.datetime.now()
    path = "logs/"+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+args.name_exp
    if not os.path.exists(path):
        os.makedirs(path)

    games = parse_games(args)
    values = [v for v in args.values.split("/")]
    for g in games :
        for v in values :
            subprocess.call(create_chpt_cmd(args, g, path, v), shell = True)
            subprocess.call(create_cmd(args, g, path, v), shell = True)
            subprocess.call(("touch "+path+"/"+g+v+"/checkpoints_saved/STOP"), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default= '', type=str, help='Name of the experiment', dest='name_exp')
    parser.add_argument('-g', default= 'po', type=str, help='Name of the games to train', dest='games')
    parser.add_argument('-p', default= 'lr', type=str, help='Name of the parameter to change', dest='param')
    parser.add_argument('-val', default='1', type=str, help='Values of the parameter', dest='values')
    parser.add_argument('-t', default= 1800, type=int, help='Period of time btw save of checkpoints', dest='time')
    parser.add_argument('--max_global_steps', default=80000000, type=int, help="Max. number of training steps", dest="max_global_steps")
    parser.add_argument('-lra', '--lr_annealing_steps', default=80000000, type=int, help="Nr. of global steps during which the learning rate will be linearly annealed towards zero", dest="lr_annealing_steps")
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
