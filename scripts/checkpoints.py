import argparse
import os
import shutil
import time

"""Pour arreter la sauvegarde des checkpoints, creer un fichier STOP
dans le dossier checkpoints_saved"""


def save_checkpoints(args, n):
    if os.path.exists( args.debugging_folder+"checkpoints"):
        pathSrc = args.debugging_folder
        pathDest = args.debugging_folder+"checkpoints_saved/"+str(n)
        if not os.path.exists(pathDest):
            os.makedirs(pathDest)
        shutil.copy((pathSrc+"/args.json"), pathDest)
        if not os.path.exists(pathDest+"/checkpoints"):
            os.makedirs(pathDest+"/checkpoints")
        for f in os.listdir(pathSrc+"/checkpoints"):
            shutil.copy((pathSrc+"/checkpoints/"+f), (pathDest+"/checkpoints"))

def main(args):
    path = args.debugging_folder+"checkpoints_saved"
    if not os.path.exists(path):
        os.makedirs(path)
    pathSTOP = args.debugging_folder+"checkpoints_saved/STOP"
    n = 0
    while not(os.path.exists(pathSTOP)):
        save_checkpoints(args, n)
        time.sleep(args.time)
        n += 1

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str,
                        help="Folder where is saved the debugging information.",
                        dest="debugging_folder")
    parser.add_argument('-t', default=1800, type=int,
                        help='Period of time btw save', dest='time')
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
