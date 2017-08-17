import argparse
import subprocess


def create_cmd(args, game_name):
    cmd = ("python3 test.py -f "+args.folder+game_name+"/"+
           " -tc "+str(args.test_count)+
           " -np "+str(args.noops)+
           " -d "+args.device)
    return cmd

def main(args):
    if('po' in args.games):
        subprocess.call(create_cmd(args, "pong"), shell = True)
    if('br' in args.games):
        subprocess.call(create_cmd(args, "breakout"), shell = True)
    if('ms' in args.games):
        subprocess.call(create_cmd(args, "ms_pacman"), shell = True)
    if('sp' in args.games):
        subprocess.call(create_cmd(args, "space_invaders"), shell = True)
    if('mo' in args.games):
        subprocess.call(create_cmd(args, "montezuma_revenge"), shell = True)
    if('se' in args.games):
        subprocess.call(create_cmd(args, "seaquest"), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', default= 'po', type=str, help='Name of the games to train', dest='games')
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    parser.add_argument('-d', '--device', default='/gpu:0', type=str, help="Device to be used ('/cpu:0', '/gpu:0', '/gpu:1',...)", dest="device")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
