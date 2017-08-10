import argparse
import subprocess
import os


def create_gif(args, n, game):
    cmd = ("python3 test.py -f "+args.folder+"checkpoints_saved/"+game+"/"+n+
           " -tc "+str(args.test_count)+
           " -np "+str(args.noops)+
           " -gn "+game+n+
           " -gf "+args.folder+"training_gifs/"+game+
           " -d "+args.device)
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
    game = parse_games(args)
    pathDest = args.folder+"training_gifs/"+game
    pathSrc = args.folder+"checkpoints_saved/"+game
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    dirs_list = os.listdir(pathSrc)
    for d in dirs_list :
        subprocess.call(create_gif(args, d, game), shell = True)
        print("###finished gif "+d)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default= 'pong', type=str,
                        help='Name of the game to train', dest='game_name')
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of the game.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
