# Scripts to simplify the use of Manette

1. Open a new shell
2. For CUDA users, run ``` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 ``` and ``` export CUDA_VISIBLE_DEVICES=0 ``` to choose the GPU to use.
3. Choose your script
4. Use nohup so that the script can continue to run when you are deconnected and the info printed out in the shell can be logged in a file : ``` nohup python3 script/NAME_SCRIPT OPTIONS &> NAME_FILE.out & ```

## BatchTrain

If you are tired of typing multiple options in the command line to use the ```train.py``` file, you can use the ```batchTrain.py``` script.
Simply write as many JSON files (like the one below) as you want, change all the options you wish and put them all in the same folder, say ```toTrain/experiment1/```.
If you want, you can have multiple folders with JSON files in  ```toTrain/```, they will also be run sequentially.

Run : ```python3 script/batchTrain -f toTrain/ -d logs/ ```.

All your JSON files will be loaded and trained, one after the other, with the right options, and saved in ```logs/DATE-experiment1/```.

Example of JSON file for Pong, with PWYX network and FiGAR 10 repetitions :
```
{
  "game": "pong",
  "initial_lr": 0.0224,
  "lr_annealing_steps": 80000000,
  "max_global_steps": 80000000,
  "max_local_steps": 5,
  "gamma": 0.99,
  "alpha": 0.99,
  "entropy_regularisation_strength": 0.02,
  "arch": "PWYX",
  "emulator_workers": 8,
  "emulator_counts": 32,
  "clip_norm_type": "global",
  "clip_norm": 3.0,
  "single_life_episodes": false,
  "e": 0.1,
  "random_start": true,
  "egreedy": false,
  "epsilon": 0.05,
  "softmax_temp": 1.0,
  "annealed": false,
  "annealed_steps": 80000000,
  "keep_percentage": 0.9,
  "rgb": false,
  "max_repetition": 11,
  "nb_choices": 10,
  "checkpoint_interval": 1000000,
  "activation": "relu",
  "alpha_leaky_relu": 0.1
}
```

 Available options :
* ``` -f ``` : Folder where to find the JSON files.
* ``` -d ``` : Folder where to save the agents.
* ``` -t ``` : Time (seconds) between each checkpoint save. Default = 1800 (30 min).

## Save checkpoints
The ```checkpoints.py``` script saves all the checkpoints (the network weights) automatically during the training. You don't have to run it, it is automatically called by ```batchTrain.py```. It is useful for the other scripts and to generate tests and gifs at different times of the training.

In the agent's folder, a new ```checkpoints_saved``` folder will be created, containing a folder for each checkpoint.

## Resume training

The ```resumeTraining.py``` script allows you to resume the training from the last checkpoint.
Run : ```python3 script/resumeTraining.py -df logs/DATE-experiment1/``` to resume the ```test_pong``` agent.
Before running the script, you can change the options in the ```args.json``` file, if you want to change the maximum of global steps or the learning rate or something in the exploration policy.

The advantage of this script is that it launches again ```checkpoints.py``` automatically : it's useful if you don't want to loose a checkpoint. Also, you don't have to type again all your training options in the command line.

## Test All
The ```testAll.py``` script allows you to test multiple agents.
Run ```python3 script/testAll.py -f logs/DATE-experiment1/ -tc 50``` to test all the agents stored in the ```logs/DATE-experiment1/``` folder, 50 times each.

**Advice** : use ```nohup python3 script/testAll.py -f logs/DATE-experiment1/ -tc 50 &> test.out``` so that the tests results are written in the ```test.out``` file.

## Generate gifs
The ```genGifs.py``` script allows you to generate gifs for multiple agents and to use the checkpoint you want.

Run ```python3 script/genGifs.py -f logs/DATE-experiment1``` to generate gifs for all the agents stored in the ```logs/``` folder.
The gifs will be stored in a ```gifs``` folder in the ```logs/``` folder.

Options :
* ``` -f ``` : Folder where the agents are stored.
* ``` -gf ``` : If you want to test only one agent, specify it's name.
* ``` -tc ``` : Number of gifs to make for each agent. Default : 1.
* ``` -cp ``` : Number of the checkpoint folder where to retrieve the network weights. Default : use the last checkpoint.


## Create Training Video
The ```createTrainingVideo.py``` script creates a video with gifs from all the checkpoints of the training, so that you can see your agent learning to play.

Run ```python3 script/createTrainingVideo.py -f logs/DATE-experiment1/pong/``` to generate the training video for the agent in the folder ```logs/DATE-experiment1/pong/```.

**Advice** : The script will use ALL the checkpoints in the ```logs/DATE-experiment1/pong/checkpoints_saved``` folder. Some checkpoints may have been saved twice. If you don't want to see the agent play twice with the same checkpoint, you should remove the checkpoint's copies from the folder.
