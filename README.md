# Modélisation des panaches de fumées industrielles — Flow Matching

Ce dépôt contient le code et les ressources pour la modélisation des panaches de fumées industrielles, basé notamment sur des approches VQGAN et Flow Matching.

Prérequis
- Conda (ou mamba) et pilotes NVIDIA installés (vérifier avec `nvidia-smi`).
- Python 3.9+

Environnement
Utilisez le fichier `env.yml` pour créer l'environnement recommandé :

```bash
# avec conda
conda env create -f env.yml -n smoke

# ou plus rapide avec mamba
mamba env create -f env.yml -n smoke

conda activate smoke
```

Installation minimale (packages utiles)

```bash
pip install requests pandas
```


Pour lancer l'entrainement Utiliser le fichier environment.yml 
```bash
conda env create -f environment.yml 

conda activate river
```


Jeu de données
Le jeu de données est un instantané de l'outil d'annotation de fumée du 24 février 2020. Il contient 12 567 clips vidéo provenant de trois sites industriels de surveillance.

Labels ciblés
Nous priorisons les annotations indiquant des émissions de fumée confirmées :
- `47` (Positive) : validation par un·e chercheur·se.
- `23` (Strong Positive) : accord de deux volontaires (ou confirmation par un·e chercheur·se).

Téléchargement des vidéos
Pour télécharger les vidéos correspondant aux labels 23 et 47, utilisez le fichier de métadonnées `metadata_02242020.json` et le script de téléchargement fourni. Les clips existent en deux résolutions :
- Standard : 180×180 px
- Haute résolution : 320×320 px (remplacer `/180/` par `/320/` dans l'URL)


Crédits et citations
Si vous utilisez ce jeu de données ou le code, merci de citer le travail Project RISE :

Yen-Chia Hsu, Ting-Hao (Kenneth) Huang, Ting-Yao Hu, Paul Dille, Sean Prendi, Ryan Hoffman, Anastasia Tsuhlares, Jessica Pachuta, Randy Sargent, Illah Nourbakhsh. 2021. "Project RISE: Recognizing Industrial Smoke Emissions." Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). https://arxiv.org/abs/2005.06111

Référence Flow Matching / VQGAN
```
@InProceedings{Davtyan_2023_ICCV,
    author    = {Davtyan, Aram and Sameni, Sepehr and Favaro, Paolo},
    title     = {Efficient Video Prediction via Sparsely Conditioned Flow Matching},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23263-23274}
}
```

Projet River
La bibliothèque "River" (référence : https://github.com/araachie/river) a été utilisée dans certaines parties du projet pour faciliter l'intégration et les expérimentations autour de VQGAN et Flow Matching. Pour des détails méthodologiques complémentaires relatifs aux approches de génération/conditionnement de flux, voir également l'article suivant : https://arxiv.org/abs/2211.14575.

## Structure du projet
Une vue d'ensemble des dossiers et scripts principaux :

- **Fichiers racine**: `dataset_init.py`, `prepare_dataset.py`, `organize_dataset.PY`, `split_labels.py`, `train.py`, `train_vqvae.py`, `README.md`, `requirements.txt`. Scripts d'aide au téléchargement, préparation et entraînement.
- **configs/**: fichiers de configuration YAML (ex. `config_vqvae.yaml`, `smoke_dataset.yaml`).
- **dataset/**: classes et utilitaires pour charger et convertir le jeu de données (`video_dataset.py`, `text_based_video_dataset.py`, `h5.py`, `convert_to_h5.py`).
- **evaluation/**: métriques et évaluateur (`evaluator.py`).
- **experimentation/**: notebooks et ressources d'expérimentation (`segmentation_methods.ipynb`, `data/`, `model/`).
- **final_dataset/**: jeu de données préparé pour l'entraînement (splits, `labels.txt`, statistiques, et sous-dossiers `train/`, `val/`, `test/`).
- **images/**: images utilisées pour la documentation ou tests rapides.
- **lutils/**: utilitaires partagés (configuration, logging, wrappers, utilitaires distribués).
- **model/**: définitions des modèles et sous-modules VQGAN (`vqgan/`), couches et blocs (`layers/`), et `model.py` principal.
- **refinement/**: scripts et modules pour l'affinage des modèles et calculs d'évaluation (SSIM, FVD, inception, etc.).
- **runs/** et **runs_vqvae/**: répertoires de sorties d'entraînement, checkpoints et reconstructions.
- **smoke_videos/**: emplacement local attendu pour vidéos brutes (si présent).
- **training/**: boucle d'entraînement, trainer et utilitaires (`trainer.py`, `training_loop.py`).

Chaque dossier contient un `__init__.py` lorsque pertinent pour faciliter l'import. Pour lancer les principaux workflows :

- Préparer le dataset : exécuter `prepare_dataset.py` puis `convert_to_h5.py` si besoin.
- Lancer un entraînement VQ-VAE : `train_vqvae.py` (voir `configs/config_vqvae.yaml`).
- Lancer l'entraînement final / flow-matching : `train.py` et les scripts dans `training/`.

Remerciements
Nous remercions le CMU CREATE Lab pour le dépôt deep-smoke-machine et pour le jeu de données public.

Licences
- Code : BSD 3-clause
- Jeu de données : Creative Commons Zero (CC0)

Fichiers utiles
- `env.yml` : configuration de l'environnement
- `metadata_02242020.json` : métadonnées pour le téléchargement des vidéos

Pour toute question ou si vous souhaitez que je reformule certaines sections en anglais ou que j'ajoute un sommaire détaillé, dites-le moi.


