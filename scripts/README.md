# Scripts pour faciliter l'utilisation de l'algorithme
Ce dossier contient plusieurs scripts permettant de faciliter l'utilisation de l'algorithme PAAC.

## Préparation préalable
* Commande pour indiquer le chemin des librairies : ``` export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 ```
* Commande pour choisir le GPU à utiliser (0, 1 ou 2) : ``` export CUDA_VISIBLE_DEVICES=2 ```
* Utiliser nohup pour que les calculs continuent meme une fois l'utilisateur deconnecté et stocker les résultats dans un fichier : ``` nohup python3 script/NOM_SCRIPT OPTIONS &> nohupLogs/NOM_FICHIER.out & ```

## Train all
Le script trainAll.py permet de lancer l'entrainement de plusieurs jeux, tout en choisissant les paramètres d'apprentissage. Les options classiques de la commande sont les suivantes :
* ``` -df ``` : debugging folder. Nom du dossier oú seront stockées les informations de debuggage. Par défaut ```logs/```.
* ``` -n ``` : Nom à donner à l'expérience. Les informations de débuggage de chaque jeu seront stockées dans le sous-dossier "Y-M-D-NOM_EXPERIENCE"
* ``` -g ``` : Noms des jeux à entrainer. Utiliser les deux premières lettres de chaque jeux. Exemple : ```-g pobrsp``` pour entrainer pong, breakout et space invaders.
* Tous les paramètres d'apprentissage peuvent aussi etre modifiés. Par défaut :
 ` --e 0.1 --alpha 0.99 -lr 0.0224 -lra 80000000 --entropy 0.02 --clip_norm 3.0 --clip_norm_type global --gamma 0.99 --max_global_steps 80000000 --max_local_steps 5 --arch NIPS --single_life_episodes False -ec 32 -ew 8 -rs True ```

## Test All
Le script testAll.py permet de lancer le test de plusieurs jeux d'affilée, une fois leur entrainement terminé. Les options classiques de la commande sont les suivantes :
* ``` -f ``` : Folder. Nom du dossier oú sont stockés les informations de debuggage de tous les jeux. Exemple : ```2017-8-10-test/```
* ``` -g ``` : Noms des jeux à tester. Utiliser les deux premières lettres de chaque jeux. Exemple : ```-g pobrsp``` pour tester pong, breakout et space invaders.
* ``` -tc ``` : Nombre de tests à effectuer pour chaque jeu. Par défaut 1.

TEST TEST

## Generate gifs
Le script genGifs.py permet de générer des gifs des jeux apès leur entrainement. Les options classiques de la commande sont les suivantes :
* ``` -f ``` : Folder. Nom du dossier oú sont stockés les informations de debuggage de tous les jeux. Exemple : ```2017-8-10-test/```. Les gifs seront stockés dans le sous-dossier ```gifs/```.
* ``` -g ``` : Noms des jeux dont on veut un gif. Utiliser les deux premières lettres de chaque jeux. Exemple : ```-g pobrsp``` pour tester pong, breakout et space invaders.
* ``` -tc ``` : Nombre de tests à effectuer pour chaque jeu. Par défaut 1.

## Save checkpoints
Le script saveCheckpoints.py permet de sauvegarder les checkpoints (ie. l'état des réseaux de neurones) au fur et à mesure de l'entrainement. Ceci permet de générer ensuite des gifs à différents moments de l'entrainement. Les options classiques de la commande sont les suivantes :
* ``` -df ``` : debugging folder. Nom du dossier oú sont stockées les informations de debuggage des jeux. Par défaut ```logs/```.
* ``` -g ``` : Noms des jeux dont on veut sauvegarder les checkpoints. Utiliser les deux premières lettres de chaque jeux. Exemple : ```-g pobrsp``` pour tester pong, breakout et space invaders.
* ``` -p ``` : période de temps entre chaque sauvegarde, en secondes. Par défaut : 1800 s = 30 min .

Dans le dossier de debuggage, un dossier ```checkpoints_saved``` sera créé, contenant un dossier pour chaque jeu, lui-meme contenant un dossier pour chaque sauvegarde.

## Create Training Gifs
Le script createTrainingGifs.py permet de créer les gifs correspondants aux sauvegardes créée par le script saveCheckpoints.py. Les options sont les memes que pour genGifs.py . Les gifs de chaque jeu seront sauvegardés dans le dossier ```training_gifs```.

## Video de l'entrainement
Il est possible de concatener tous les gifs d'un entrainement pour en faire une video. Pour cela il faut se placer sur une machine où ffmpeg et mkvmerge sont installés. On pourra alors utiliser les scripts makeVideo.py et mergeVideos.py .
