import argparse
import subprocess
import os


def create_cmd(args, f):
    cmd = ("python3 test.py -f "+args.folder+f+"/"+
           " -tc "+str(args.test_count)+
           " -np "+str(args.noops))
    return cmd

def main(args):
    for f in os.listdir(args.folder):
        subprocess.call(create_cmd(args, f), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the debugging information.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
