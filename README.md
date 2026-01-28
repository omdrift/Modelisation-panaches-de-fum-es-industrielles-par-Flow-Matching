# Modélisation des panaches de fumées industrielles — Flow Matching

Ce dépôt contient le code et les ressources pour la modélisation des panaches de fumées industrielles, basé notamment sur des approches VQGAN et Flow Matching.
## Quelques exemples de travaux

- Vidéo originale (résolution standard 180×180) :
  (6_0-0-2018-06-11-6304-964-6807-1467-180-180-3470-1528712115-1528712290.mp4)
<table border="0">
  <tr>
    <td>
      <p align="center"><b>Source (180x180)</b></p>
      <video src="media/6_0-0-2018-06-11-6304-964-6807-1467-180-180-3470-1528712115-1528712290.mp4" width="180" controls></video>
    </td>
    <td>
      <p align="center"><b>Reconstruction VQ-VAE</b></p>
      <img src="media/custom_recon_epoch_56.png" width="250" alt="Reconstruction VQ-VAE">
    </td>
  </tr>
</table>

- Extraction et traitement des masques de segmentation pour l'entraînement du modèle.
<p align="center">
  <img src="media/6_0-0-2018-06-11-6304-964-6807-1467-180-180-3470-1528712115-1528712290_frame0001.png" width="180" alt="Segmentation Frame">
</p>
  
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
- **configs/**: fichiers de configuration YAML utilisés pour les entraînements et la préparation des modèles. Exemples principaux :
    - `configs/config_vqvae.yaml` : configuration du VQ-VAE / VQGAN (architecture de l'encodeur/décodeur, paramètres du vector-quantizer, embedding dim, etc.). Utilisé par `train_vqvae.py` via l'argument `--config`.
    - `configs/smoke_dataset.yaml` : configuration pour le training Flow Matching (chemins `data.data_root`, paramètres du modèle `vector_field_regressor`, hyperparamètres d'entraînement, évaluation). Utilisé par `train.py` pour lancer l'entraînement final.
    - Autres YAML : ajoutez vos propres fichiers de config ici pour expérimentations (ex. variantes d'architecture ou chemins locaux).

    Utilisation rapide : passez `--config configs/config_vqvae.yaml` à `train_vqvae.py` et `--config configs/smoke_dataset.yaml` à `train.py`. Modifiez les chemins (`data.data_root`, checkpoints) directement dans les YAML pour pointer vers vos dossiers locaux.
- **dataset/**: classes et utilitaires pour charger et convertir le jeu de données (`video_dataset.py`, `text_based_video_dataset.py`, `h5.py`, `convert_to_h5.py`).

### Scripts pour préparer le jeu de données final
Les scripts suivants permettent de construire le dossier `final_dataset/` à partir des vidéos brutes et des métadonnées :

- `dataset_init.py` : télécharge les vidéos correspondant aux labels ciblés (23 et 47) depuis `metadata_02242020.json` vers le dossier `smoke_videos/`. Modifier les constantes en haut du fichier pour changer la résolution, le dossier de destination ou le nombre de threads. Exemple :

```bash
python dataset_init.py
```

- `prepare_dataset.py` : extrait les images depuis les vidéos (`smoke_videos/`) et applique un pré-traitement / matting pour isoler les panaches de fumée ; écrit les frames dans `isolated_smoke_frames/`.

```bash
python prepare_dataset.py
```

- `organize_dataset.PY` : regroupe les frames extraites par vidéo, renomme pour éviter collisions, et crée le split `final_dataset/train/` et `final_dataset/test/` ainsi que les fichiers `train_files.txt`, `test_files.txt` et `labels.txt`.

```bash
python organize_dataset.PY
```

- `dataset/convert_to_h5.py` (ou `convert_to_h5.py` racine) : convertit les images et listes en un fichier HDF5 optimisé pour les dataloaders (si présent). Utiliser si vous préférez charger les données depuis un seul fichier h5.

- `split_labels.py` : vérifie l'intégrité des séquences (frames manquantes) et génère `missing_frames_report.json` si nécessaire.

- `dataset/text_based_video_dataset.py` : classe `TextBasedVideoDataset` utilisée par `train_vqvae.py` et `train.py` pour charger les séquences en se basant sur `train_files.txt` / `val_files.txt`.

Workflow recommandé (exécution séquentielle) :

```bash
python dataset_init.py        # télécharger vidéos sélectionnées
python prepare_dataset.py     # extraire et isoler frames
python organize_dataset.PY    # organiser, renommer et créer splits
python split_labels.py        # vérifier les frames manquantes
```

Modifier les chemins et paramètres directement dans les scripts ou via les fichiers de configuration YAML selon vos besoins.

## Lancer l'entraînement VQ-VAE (VQGAN)
Suivez ces étapes pour entraîner le VQ-VAE (utilisé ici comme VQGAN) sur le `final_dataset`.

1) Créer et activer l'environnement

```bash
conda env create -f environment.yml -n river
conda activate river
```


3) Préparer le dataset (si pas déjà fait)

```bash
python dataset_init.py
python prepare_dataset.py
python organize_dataset.PY
python split_labels.py
```

4) Vérifier que `final_dataset/train_files.txt` et `final_dataset/val_files.txt` existent et que `final_dataset/train/` contient les images.

5) Lancer l'entraînement VQ-VAE

Exemple d'appel (ajustez `--batch-size`, `--epochs`, `--num-workers` selon votre GPU/CPU) :

```bash
python train_vqvae.py \
    --config configs/config_vqvae.yaml \
    --run-name smoke_vqvae_experiment \
    --batch-size 8 \
    --num-workers 4 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb
```

- Sorties : dossier `runs/vqvae_smoke_vqvae_experiment/` contenant `reconstructions/` et checkpoints (selon implémentation).
- Pour tester une image fixe à chaque époque, ajoutez `--test-image path/to/image.png`.

6) Relancer / reprendre l'entraînement

Si votre script sauvegarde des checkpoints, restaurez en adaptant le code de chargement ou en passant un argument de reprise (selon implémentation). Pour le workflow Flow Matching complet, utilisez `train.py` (voir configs/smoke_dataset.yaml) :

```bash
python train.py --run-name flow_run --config configs/smoke_dataset.yaml --num-gpus 1 --wandb
```

Remarques
- Assurez-vous que CUDA est disponible (`nvidia-smi`) et que `torch.cuda.is_available()` renvoie True.
- Ajustez `configs/config_vqvae.yaml` pour modifier architecture, taille d'embedding ou hyperparamètres.
- Pour le suivi d'expérimentation, activez `--wandb` et configurez votre compte Weights & Biases.

Section terminée. 
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



