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
                " --single_life_episodes "+str(data["single_life_episodes"])+
                " -ec "+str(data["emulator_counts"])+
                " -ew "+str(data["emulator_workers"])+
                " -rs "+str(data["random_start"])+

                " --egreedy "+str(data["egreedy"])+
                " --epsilon "+str(data["epsilon"])+
                " --softmax_temp "+str(data["softmax_temp"])+
                " --annealed "+str(data["annealed"])+
                " --keep_percentage "+str(data["keep_percentage"])+
                " --rgb "+str(data["rgb"])+
                " --random_actions "+str(data["random_actions"])+
                " --nb_actions "+str(data["nb_actions"])+
                " --oxygen_greedy "+str(data["oxygen_greedy"])+
                " --proba_oxygen "+str(data["proba_oxygen"])+
                " --nb_up_actions "+str(data["nb_up_actions"])+
                " --FiGAR "+str(data["FiGAR"])+
                " --max_repetition "+str(data["max_repetition"]))
    return cmd

def create_chpt_cmd(args, path):
    cmd = ("nohup python3 scripts/checkpoints.py "+
                " -df "+path+"/"
                " -t "+str(args.time)+
                " &> nohupLogs/saveCheckpoints"+str(args.gpu)+".out &")
    return cmd


def main(args):
    pathSrc = "toTrain/gpu"+str(args.gpu)
    for folder in os.listdir(pathSrc):
        i = datetime.datetime.now()
        #path = "logs/"+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+folder
        path = "/data1/rl/atari/logs/tensorboard/"+str(i.year)+"-"+str(i.month)+"-"+str(i.day)+"-"+folder
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
    parser.add_argument('-gpu', default= 0, type=int,
                        help='Number of the gpu to be used', dest='gpu')
    parser.add_argument('-t', default=1800, type=int,
                        help='Period of time btw save', dest='time')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
