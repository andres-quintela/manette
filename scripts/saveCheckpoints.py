import argparse
import os
import shutil
import time

"""Pour arreter la sauvegarde des checkpoints, creer un fichier STOP
dans le dossier checkpoints_saved"""

def game_dirs(args, game_name):
    path = args.debugging_folder+"checkpoints_saved"
    pathDest = path+"/"+game_name
    pathGameCP = path+"/"+game_name+"/checkpoints"
    pathSrc = args.debugging_folder+game_name+"/args.json"
    if not os.path.exists(pathGameCP):
        os.makedirs(pathGameCP)
    if os.path.exists(pathSrc):
        shutil.copy(pathSrc, pathDest)

def init_dirs(args):
    path = args.debugging_folder+"checkpoints_saved"
    if not os.path.exists(path):
        os.makedirs(path)
    if('po' in args.games): game_dirs(args, "pong")
    if('br' in args.games): game_dirs(args, "breakout")
    if('ms' in args.games): game_dirs(args, "ms_pacman")
    if('sp' in args.games): game_dirs(args, "space_invaders")
    if('mo' in args.games): game_dirs(args, "montezuma_revenge")
    if('se' in args.games): game_dirs(args, "seaquest")

def save_checkpoints(args, game_name):
    if os.path.exists( args.debugging_folder+"/"+game_name):
        pathSrc = args.debugging_folder+"/"+game_name+"/checkpoints"
        pathDest = args.debugging_folder+"checkpoints_saved/"+game_name+"/checkpoints"
        for f in os.listdir(pathSrc):
            shutil.copy((pathSrc+"/"+f), pathDest)

def main(args):
    init_dirs(args)
    pathSTOP = args.debugging_folder+"checkpoints_saved/STOP"
    while not(os.path.exists(pathSTOP)):
        init_dirs(args)
        if('po' in args.games): save_checkpoints(args, "pong")
        if('br' in args.games): save_checkpoints(args, "breakout")
        if('ms' in args.games): save_checkpoints(args, "ms_pacman")
        if('sp' in args.games): save_checkpoints(args, "space_invaders")
        if('mo' in args.games): save_checkpoints(args, "montezuma_revenge")
        if('se' in args.games): save_checkpoints(args, "seaquest")
        time.sleep(1800)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where is saved the debugging information.",
                        dest="debugging_folder")
    parser.add_argument('-g', default= 'po', type=str,
                        help='Names of the games to train', dest='games')
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
