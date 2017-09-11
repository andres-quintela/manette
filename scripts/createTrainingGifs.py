import argparse
import subprocess
import os


def create_gif(args, n):
    cmd = ("python3 test.py -f "+args.folder+"checkpoints_saved/"+n+
           " -tc "+str(args.test_count)+
           " -np "+str(args.noops)+
           " -gn "+n+
           " -gf "+args.folder+"training_gifs/")
    return cmd


def main(args):
    pathDest = args.folder+"training_gifs/"
    pathSrc = args.folder+"checkpoints_saved/"
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    dirs_list = os.listdir(pathSrc)
    for d in dirs_list :
        subprocess.call(create_gif(args, d), shell = True)
        print("###finished gif "+d)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of the game.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
