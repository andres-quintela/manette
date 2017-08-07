import shutil
import os
import argparse
import subprocess


def main(args):
    dest_dir = "/data1/rl/atari/logs"
    shutil.move(args.source_dir, dest_dir)
    print("Directory "+args.source_dir+" archived." )


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', default='logs/', type=str,
                        help='Name of the source directory. Default : all the logs directory is moved.',
                        dest='source_dir')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
