import subprocess
import os
import argparse

"""Must be launched on laptop, not on eris"""

def create_cmd_convert(n, path, game):
    gif_name = path+game+str(n)+'0.gif'
    video_name = path+str(n)+".mp4"
    cmd = ('ffmpeg -i '+gif_name+' -movflags faststart -pix_fmt yuv420p'+
           ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '+video_name)
    return cmd

def main(args):
    game = parse_games(args)
    path = "../../gifs/trainingGifs/"+args.folder
    if not os.path.exists(path):
        os.makedirs(path)
    f_list = os.listdir(path)
    name_list = []
    for f in f_list :
        name_list.append(int(f[len(game):-5]))
    print(name_list)
    for n in name_list:
        print(n)
        subprocess.call(create_cmd_convert(n, path, game), shell = True)
    #subprocess.call(create_cmd_merge(min(name_list), max(name_list), path, game), shell = True)

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
