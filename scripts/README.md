# Scripts pour faciliter l'utilisation de l'algorithme
Ce dossier contient plusieurs scripts permettant de faciliter l'utilisation de l'algorithme PAAC.

# Train all
Le script trainAll.py permet de lancer l'entrainement de plusieurs jeux, tout en choisissant les paramètres d'apprentissage. Les options classiques de la commande sont les suivantes :
* ``` -df ``` : debugging folder. Nom du dossier oú seront stockées les informations de debuggage. Par défaut "logs/".
* ``` -n ``` : Nom à donner à l'expérience. Les informations de débuggage de chaque jeu seront stockées dans le sous-dossier "Y-M-D-NOM_EXPERIENCE"
* ``` -g ``` : Noms des jeux à entrainer. Utiliser les deux premières lettres de chaque jeux. Exemple : "-g pobrsp" pour entrainer pong, breakout et space invaders.
* Tous les paramètres d'apprentissage peuvent aussi etre modifiés. Par défaut :
``` --e 0.1 --alpha 0.99 -lr 0.0224 -lra 80000000 --entropy 0.02 --clip_norm 3.0 --clip_norm_type global --gamma 0.99 --max_global_steps 80000000 --max_local_steps 5 --arch NIPS --single_life_episodes False -ec 32 -ew 8 -rs True ```
