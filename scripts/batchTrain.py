import argparse
import os
import datetime
import json
import subprocess

def create_cmd(data, path):
    cmd = ("python3 train.py -g "+data["game"]+" -df "+path+"/"+
                " --e "+str(data["e"])+
                " --alpha "+str(data["alpha"])+
                " -lr "+str(data["initial_lr"])+
                " -lra "+str(data["lr_annealing_steps"])+
                " --entropy "+str(data["entropy_regularisation_strength"])+
                " --clip_norm "+str(data["clip_norm"])+
                " --clip_norm_type "+str(data["clip_norm_type"])+
                " --gamma "+str(data["gamma"])+
                " --max_global_steps "+str(data["max_global_steps"])+
                " --max_local_steps "+str(data["max_local_steps"])+
                " --arch "+str(data["arch"])+
                " -ec "+str(data["emulator_counts"])+
                " -ew "+str(data["emulator_workers"])+
                " --epsilon "+str(data["epsilon"])+
                " --softmax_temp "+str(data["softmax_temp"])+
                " --annealed_steps "+str(data["annealed_steps"])+
                " --keep_percentage "+str(data["keep_percentage"])+
                " --max_repetition "+str(data["max_repetition"])+
                " --nb_choices "+str(data["nb_choices"])+
                " --checkpoint_interval "+str(data["checkpoint_interval"])+
                " --activation "+str(data["activation"])+
                " --alpha_leaky_relu "+str(data["alpha_leaky_relu"]))
    if data["single_life_episodes"] : cmd += " --single_life_episodes"
    if data["random_start"] : cmd += " --random_start"
    if data["egreedy"] : cmd += " --egreedy"
    if data["annealed"] : cmd += " --annealed"
    if data["rgb"] : cmd += " --rgb"
    return cmd

def create_chpt_cmd(args, path):
    cmd = ("nohup python3 scripts/checkpoints.py "+
                " -df "+path+"/"
                " -t "+str(args.time)+
                " &> nohupLogs/saveCheckpoints.out &")
    return cmd


def main(args):
    pathSrc = args.folder
    for folder in os.listdir(pathSrc):
        i = datetime.datetime.now()
        path = args.destination+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+folder
        if not os.path.exists(path):
            os.makedirs(path)
        for f in os.listdir(pathSrc+"/"+folder):
            with open(pathSrc+"/"+folder+"/"+f, 'r') as d :
                data = json.load(d)
                pathDest = path + "/"+f[:-5]
                subprocess.call(create_chpt_cmd(args, pathDest), shell = True)
                subprocess.call(create_cmd(data, pathDest), shell = True)
                subprocess.call(("touch "+pathDest+"/checkpoints_saved/STOP"), shell = True)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', default='toTrain', type=str,
                        help='Folder where to find the JSON files with the training options', dest='folder')
    parser.add_argument('-t', default=1800, type=int,
                        help='Period of time btw checkpoints save', dest='time')
    parser.add_argument('-d', default='logs/', type=str,
                        help='Folder where to save the training information', dest='destination')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
