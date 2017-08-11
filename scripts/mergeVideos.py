import subprocess
import os
import argparse

"""Must be launched on laptop, not on eris"""

def create_cmd_merge(name_list, path, game):
    output_name = path+game+"Training.mp4"
    cmd = ("mkvmerge -o "+output_name)
    l = name_list
    l.sort()
    print(l)
    for n in l[:-1]:
        cmd += " "+path+str(n)+".mp4 \+ "+path+"black.mp4 \+"
    cmd += " "+path+str(l[-1])+".mp4"
    return cmd

def main(args):
    game = parse_games(args)
    path = "../../gifs/trainingGifs/"+args.folder
    if not os.path.exists(path):
        os.makedirs(path)
    f_list = os.listdir(path)
    name_list = []
    for f in f_list :
        if (f != "black.mp4" and f[-3:]=="mp4"):
            name_list.append(int(f[:-4]))
    print(name_list)
    cmd = create_cmd_merge(name_list, path, game)
    print(cmd)
    subprocess.call(cmd, shell = True)

def parse_games(args):
    game = ""
    if('po' in args.games): game = "pong"
    if('br' in args.games): game = "breakout"
    if('ms' in args.games): game = "ms_pacman"
    if('sp' in args.games): game = "space_invaders"
    if('mo' in args.games): game = "montezuma_revenge"
    if('se' in args.games): game = "seaquest"
    return game

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default= 'po', type=str, help='Name of the games to train', dest='games')
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of all games.", dest="folder", required=True)
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
