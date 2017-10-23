import argparse
import subprocess
import os


def create_cmd(args, f, path):
    cmd = ("python3 test.py -f "+path+
           " -tc "+str(args.test_count)+
           " -np "+str(args.noops)+
           " -gn "+f+str(args.checkpoint)+
           " -gf "+args.folder+"gifs/")
    return cmd

def main(args):
    path = args.folder+"gifs"
    if not os.path.exists(path):
        os.makedirs(path)
    if args.game_folder == '' :
        print(os.listdir(args.folder))
        for f in os.listdir(args.folder):
            if f != "gifs":
                if args.checkpoint == 0 :
                    pathSrc = args.folder+f
                else :
                    pathSrc = args.folder+f+"/checkpoints_saved/"+str(args.checkpoint)+"/"
                subprocess.call(create_cmd(args, f, pathSrc), shell = True)
    else :
        f = args.game_folder
        if args.checkpoint == 0 :
            pathSrc = args.folder+f
        else :
            pathSrc = args.folder+f+"checkpoints_saved/"+str(args.checkpoint)+"/"
        subprocess.call(create_cmd(args, f, pathSrc), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gf', default= '', type=str, help='Name of the game folder to generate. Default : all folder are generated.', dest='game_folder')
    parser.add_argument('-f', '--folder', type=str, help="Folder where is saved the logs of all games.", dest="folder", required=True)
    parser.add_argument('-tc', '--test_count', default='1', type=int, help="The amount of tests to run on the given network", dest="test_count")
    parser.add_argument('-cp', '--checkpoint', default='0', type=int, help="The checkpoint from which to run the test", dest="checkpoint")
    parser.add_argument('-np', '--noops', default=30, type=int, help="Maximum amount of no-ops to use", dest="noops")
    return parser


if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
