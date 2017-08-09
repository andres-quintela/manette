import argparse
import os
import shutil
import time

"""Pour arreter la sauvegarde des checkpoints, creer un fichier STOP
dans le dossier checkpoints_saved"""

def init_dirs(args, games):
    path = args.debugging_folder+"checkpoints_saved"
    if not os.path.exists(path):
        os.makedirs(path)
    for game_name in games :
        pathGame = args.debugging_folder+"checkpoints_saved/"+game_name
        if not os.path.exists(pathGame):
            os.makedirs(pathGame)

def save_checkpoints(args, game_name, n):
    if os.path.exists( args.debugging_folder+game_name):
        pathSrc = args.debugging_folder+game_name
        pathDest = args.debugging_folder+"checkpoints_saved/"+game_name+"/"+str(n)
        if not os.path.exists(pathDest):
            os.makedirs(pathDest)
        shutil.copy((pathSrc+"/args.json"), pathDest)
	if not os.path.exists(pathDest+"/checkpoints"):
	    os.makedirs(pathDest+"/checkpoints")
	for f in os.listdir(pathSrc+"checkpoints"):
            shutil.copy((pathSrc+"/checkpoints/"+f), (pathDest+"/checkpoints"))

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
    games = parse_games(args)
    init_dirs(args, games)
    pathSTOP = args.debugging_folder+"checkpoints_saved/STOP"
    n = 0
    while not(os.path.exists(pathSTOP)):
        init_dirs(args, games)
        for g in games :
            save_checkpoints(args, g, n)
        time.sleep(args.period)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where is saved the debugging information.",
                        dest="debugging_folder")
    parser.add_argument('-g', default= 'po', type=str,
                        help='Names of the games to train', dest='games')
    parser.add_argument('-p', default=1800, type=int,
                        help='Period of time btw save', dest='period')
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
