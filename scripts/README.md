# Scripts pour faciliter l'utilisation de l'algorithme
Ce dossier contient plusieurs scripts permettant de faciliter l'utilisation de l'algorithme PAAC.

## Préparation préalable
Dans un nouveau shell, tapez :
* Commande pour indiquer le chemin des librairies : ``` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 ```
* Commande pour choisir le GPU à utiliser (0, 1 ou 2) : ``` export CUDA_VISIBLE_DEVICES=2 ```
* Utiliser nohup pour que les calculs continuent meme une fois l'utilisateur deconnecté et stocker les résultats dans un fichier : ``` nohup python3 script/NOM_SCRIPT OPTIONS &> nohupLogs/NOM_FICHIER.out & ```

## BatchTrain
Le script batchTrain.py permet de lancer plusieurs entrainements sur plusieurs gpu et de facon séquentielle. Pour cela, placez dans le dossier toTrain/gpuNUM_GPU un ou plusieurs dossiers contenant les fichiers .json indiquant les paramètres des entrainements que vous voulez lancer. Exemple : toTrain/gpu0/experiment/pong_low_lr.json . Les resultats de l'entrainement seront alors stocké dans logs/Y-M-D-experiment/pong_low_lr/ . Options de la commande :
* ``` -gpu ``` : Numero du gpu utilisé pour cet entrainement.
* ``` -t ``` : Temps en seconde entre chaque sauvegarde de checkpoints. Par défaut 1800, ie 30 min.

## Test All
Le script testAll.py permet de lancer le test de plusieurs jeux d'affilée, une fois leur entrainement terminé. Les options classiques de la commande sont les suivantes :
* ``` -f ``` : Folder. Nom du dossier oú sont stockées les informations de debuggage de tous les jeux. Exemple : ```logs/2017-8-10-test/```
* ``` -tc ``` : Nombre de tests à effectuer pour chaque jeu. Par défaut 1.

## Generate gifs
Le script genGifs.py permet de générer des gifs des jeux apès leur entrainement. Les options classiques de la commande sont les suivantes :
* ``` -f ``` : Folder. Nom du dossier oú sont stockés les informations de debuggage de tous les jeux. Exemple : ```logs/2017-8-10-test/```. Les gifs seront stockés dans le sous-dossier ```gifs/```.
* ``` -gf ``` : Nom du jeu/experience dont on veut un gif en particulier. Cela correspond aux noms du sous-dossier. Si rien n'est précisé tous les gifs seront générés.
* ``` -tc ``` : Nombre de tests à effectuer pour chaque jeu. Par défaut 1.
* ``` -cp ``` : Numéro du dossier de checkpoints_saved/ avec lequel on veut générer le gifs. Si rien n'est précisé, le gif est généré avec le dernier checkpoint de l'entrainement.

## Save checkpoints
Le script saveCheckpoints.py permet de sauvegarder les checkpoints (ie. l'état des réseaux de neurones) au fur et à mesure de l'entrainement. Ceci permet de générer ensuite des gifs à différents moments de l'entrainement. Ce script est appelé automatiquement par batchTrain.py .

Dans le dossier de debuggage du jeu, un dossier ```checkpoints_saved``` sera créé,  contenant un dossier pour chaque sauvegarde de checkpoint.

## Create Training Gifs
Le script createTrainingGifs.py permet de créer les gifs de tout l'entrainement d'un jeu,  correspondants aux sauvegardes créée par le script saveCheckpoints.py. Les options sont les memes que pour genGifs.py . Les gifs de chaque jeu seront sauvegardés dans le dossier ```training_gifs```.

## Video de l'entrainement
Il est possible de concatener tous les gifs d'un entrainement pour en faire une video. Pour cela il faut se placer sur une machine où ffmpeg et mkvmerge sont installés. On pourra alors utiliser les scripts makeVideo.py et mergeVideos.py .
