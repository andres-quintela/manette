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
                " -df "+path+"/"+
                " -t "+str(args.time)+
                " &> nohupLogs/saveCheckpoints.out &")
    return cmd

def main(args):
    pathJson = args.debugging_folder+"args.json"
    with open(pathJson, 'r') as d :
        data = json.load(d)
        subprocess.call(create_chpt_cmd(args, args.debugging_folder), shell = True)
        subprocess.call(create_cmd(data, args.debugging_folder), shell = True)
        subprocess.call(("touch "+args.debugging_folder+"/checkpoints_saved/STOP"), shell = True)

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--debugging_folder', default='logs/', type=str, help="Folder where to save the debugging information.", dest="debugging_folder")
    parser.add_argument('-t', default=1800, type=int,
                        help='Period of time btw save', dest='time')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    main(args)
