import argparse
import subprocess
import os


def create_gif(args, n):
    cmd = ("python3 test.py -f "+args.folder+"checkpoints_saved/"+n+
           " -gn "+n+
           " -gf "+args.folder+"training_gifs/")
    return cmd

def create_cmd_convert(n, pathSrc, pathDest):
    gif_name = pathSrc+str(n)+'0.gif'
    video_name = pathDest+str(n)+".mp4"
    cmd = ('ffmpeg -i '+gif_name+' -movflags faststart -pix_fmt yuv420p'+
           ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '+video_name)
    return cmd

def create_cmd_merge(name_list, dest, path):
    output_name = dest+"Training.mp4"
    cmd = ("mkvmerge -o "+output_name)
    l = name_list
    l.sort()
    print(l)
    for n in l[:-1]:
        cmd += " "+path+str(n)+".mp4 \+"
    cmd += " "+path+str(l[-1])+".mp4"
    return cmd

def main(args):
    # make the gifs for all the checkpoints in checkpoints_saved
    pathDest = args.folder+"training_gifs/"
    pathSrc = args.folder+"checkpoints_saved/"
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    dirs_list = os.listdir(pathSrc)
    dirs_list.sort()
    for d in dirs_list :
        subprocess.call(create_gif(args, d), shell = True)
        print("----- Finished gif "+d+" / "+str(len(dirs_list)))

    # convert all the gifs to mp4
    pathSrc = args.folder+"training_gifs/"
    pathDest = args.folder+"training_videos/"
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    f_list = os.listdir(pathSrc)
    name_list = []
    for f in f_list :
        name_list.append(int(f[:-5]))
    name_list.sort()
    for n in name_list:
        subprocess.call(create_cmd_convert(n, pathSrc, pathDest), shell = True)
        print('----- Converted : '+str(n)+" / "+str(len(name_list)))

    # concatenate all the videos
    cmd = create_cmd_merge(name_list, args.folder, pathDest)
    print(cmd)
    subprocess.call(cmd, shell = True)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of the game.", dest="folder", required=True)
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
