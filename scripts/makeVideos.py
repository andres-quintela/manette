import subprocess
import os
import argparse


def create_cmd_convert(n, path, game):
    gif_name = path+str(n)+'0.gif'
    video_name = path+str(n)+".mp4"
    cmd = ('ffmpeg -i '+gif_name+' -movflags faststart -pix_fmt yuv420p'+
           ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '+video_name)
    return cmd

def main(args):
    pathSrc = args.folder+"trainingGifs/"
    pathDest = args.folder+"trainingVideos/"
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    f_list = os.listdir(pathSrc)
    name_list = []
    for f in f_list :
        name_list.append(int(f[:-5]))
    for n in name_list:
        print(str(n)+" / "+str(len(name_list)))
        subprocess.call(create_cmd_convert(n, pathSrc), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of all games.", dest="folder", required=True)
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
