# Modélisation des panaches de fumées industrielles — Flow Matching

Ce dépôt contient le code et les ressources pour la modélisation des panaches de fumées industrielles, basé notamment sur des approches VQGAN et Flow Matching.

## Quelques exemples de travaux

### Reconstruction VQGAN
Comparaison entre les images originales et leurs reconstructions par le VQGAN :

<p align="center">
  <img src="vqgan_reconstruction_comparison.png" width="800" alt="Comparaison reconstruction VQGAN">
</p>

### Génération de séquences vidéo
Résultats de génération de séquences de fumée avec le modèle Flow Matching :

<p align="center">
  <img src="figure5_generation.png" width="800" alt="Génération de séquences">
</p>

### Comparaison vidéos réelles vs générées

<table align="center">
  <tr>
    <td align="center">
      <b>Vidéos Réelles</b><br>
      <img src="wandb/latest-run/files/media/videos/Training/Media/real_videos_109999_c99bc7aa458f1f121210.gif" width="300" alt="Vidéos réelles">
    </td>
    <td align="center">
      <b>Vidéos Générées</b><br>
      <img src="wandb/latest-run/files/media/videos/Training/Media/generated_videos_109999_f96b7834ed7ac1939875.gif" width="300" alt="Vidéos générées">
    </td>
  </tr>
</table>

---

## Prérequis

### Matériel
- GPU NVIDIA avec au moins 8 GB de VRAM (recommandé : 16 GB ou plus)
- 32 GB de RAM système recommandés
- Espace disque : ~50 GB pour le dataset + ~20 GB pour les checkpoints

### Logiciels
- Conda (ou mamba) et pilotes NVIDIA installés (vérifier avec `nvidia-smi`)
- Python 3.9+
- CUDA 11.8+ (compatible avec PyTorch 2.0+)

## Installation de l'environnement

### 1. Cloner le dépôt
```bash
git clone https://github.com/omdrift/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching.git
cd Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching
```

### 2. Créer l'environnement Conda
Utilisez le fichier `environment.yml` pour créer l'environnement recommandé :

```bash
conda env create -f environment.yml -n river
conda activate river
```

### 3. Vérifier l'installation
```bash
# Vérifier CUDA
nvidia-smi

# Vérifier PyTorch et CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Vérifier les packages principaux
python -c "import torchvision, wandb, opencv; print('Tous les packages sont installés correctement')"
```

---

## Dataset : Préparation complète

### Vue d'ensemble
Le jeu de données est un instantané de l'outil d'annotation de fumée du 24 février 2020. Il contient 12 567 clips vidéo provenant de trois sites industriels de surveillance.

### Labels ciblés
Nous priorisons les annotations indiquant des émissions de fumée confirmées :
- `47` (Positive) : validation par un·e chercheur·se.
- `23` (Strong Positive) : accord de deux volontaires (ou confirmation par un·e chercheur·se).

### Téléchargement des vidéos
Pour télécharger les vidéos correspondant aux labels 23 et 47, utilisez le fichier de métadonnées `metadata_02242020.json` et le script de téléchargement fourni. Les clips existent en deux résolutions :
- Standard : 180×180 px
- Haute résolution : 320×320 px (remplacer `/180/` par `/320/` dans l'URL)

### Scripts de préparation du dataset

Le pipeline complet de préparation du dataset se décompose en 4 étapes séquentielles :

#### **Étape 1 : Téléchargement des vidéos** (`dataset_init.py`)

Ce script télécharge les vidéos depuis les URLs du fichier `metadata_02242020.json`.

```bash
python dataset_init.py
```

**Options de configuration** (modifiables dans le script) :
- `RESOLUTION` : `"180"` ou `"320"` pour la résolution des vidéos
- `OUTPUT_DIR` : dossier de destination (défaut : `smoke_videos/`)
- `MAX_WORKERS` : nombre de threads pour téléchargement parallèle (défaut : 10)
- `LABELS` : liste des labels à télécharger (défaut : `[23, 47]`)

**Sortie** : Dossier `smoke_videos/` contenant les vidéos organisées par vue (ex: `view_0-10/`, `view_1-4/`)

#### **Étape 2 : Extraction et matting des frames** (`prepare_dataset.py`)

Extrait les frames de chaque vidéo et applique un algorithme de matting pour isoler les panaches de fumée.

```bash
python prepare_dataset.py
```

**Paramètres importants** (dans le script) :
- `INPUT_DIR` : dossier des vidéos source (`smoke_videos/`)
- `OUTPUT_DIR` : dossier de sortie (`isolated_smoke_frames/`)
- `BACKGROUND_MODEL` : chemin vers le modèle RVM (`rvm_resnet50.pth`)
- `NUM_WORKERS` : parallélisation du traitement

**Processus** :
1. Charge chaque vidéo MP4
2. Extrait les frames individuelles
3. Applique le matting RVM (Robust Video Matting) pour isoler la fumée
4. Sauvegarde les frames isolées en PNG

**Sortie** : Dossier `isolated_smoke_frames/` avec structure miroir de `smoke_videos/`

#### **Étape 3 : Organisation et split du dataset** (`organize_dataset.py`)

Organise les frames extraites en séquences et crée les splits train/val/test.

```bash
python organize_dataset.py
```

**Fonctionnalités** :
- Regroupe les frames par séquence vidéo
- Renomme les fichiers avec un format standardisé : `video_XXXX_frame_YYYY.png`
- Crée automatiquement les splits : 80% train / 10% val / 10% test
- Génère les fichiers de liste : `train_files.txt`, `val_files.txt`, `test_files.txt`
- Calcule les statistiques du dataset (`dataset_stats.json`)

**Sortie** : 
```
final_dataset/
├── train/               # Frames d'entraînement
├── val/                 # Frames de validation
├── test/                # Frames de test
├── train_files.txt      # Liste des séquences d'entraînement
├── val_files.txt        # Liste des séquences de validation
├── test_files.txt       # Liste des séquences de test
└── dataset_stats.json   # Statistiques du dataset
```

#### **Étape 4 : Vérification d'intégrité** (`split_labels.py`)

Vérifie que toutes les séquences sont complètes (pas de frames manquantes).

```bash
python split_labels.py
```

**Sortie** : Fichier `missing_frames_report.json` si des frames sont manquantes

### Workflow complet recommandé

Exécutez ces commandes séquentiellement :

```bash
# 1. Télécharger les vidéos (peut prendre plusieurs heures)
python dataset_init.py

# 2. Extraire et isoler les frames (traitement intensif)
python prepare_dataset.py

# 3. Organiser le dataset et créer les splits
python organize_dataset.py

# 4. Vérifier l'intégrité
python split_labels.py

# 5. Vérifier la structure finale
ls -lh final_dataset/
cat final_dataset/dataset_stats.json
```

**Durée estimée** : 4-8 heures selon votre connexion internet et CPU/GPU

---

---

## Crédits et citations

### Project RISE
Si vous utilisez ce jeu de données ou le code, merci de citer le travail Project RISE :

```bibtex
@inproceedings{hsu2021project,
  title={Project RISE: Recognizing Industrial Smoke Emissions},
  author={Hsu, Yen-Chia and Huang, Ting-Hao (Kenneth) and Hu, Ting-Yao and Dille, Paul and Prendi, Sean and Hoffman, Ryan and Tsuhlares, Anastasia and Pachuta, Jessica and Sargent, Randy and Nourbakhsh, Illah},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021},
  url={https://arxiv.org/abs/2005.06111}
}
```

### Flow Matching / VQGAN
```bibtex
@InProceedings{Davtyan_2023_ICCV,
    author    = {Davtyan, Aram and Sameni, Sepehr and Favaro, Paolo},
    title     = {Efficient Video Prediction via Sparsely Conditioned Flow Matching},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23263-23274}
}
```

### Projet River
La bibliothèque "River" (référence : https://github.com/araachie/river) a été utilisée dans certaines parties du projet pour faciliter l'intégration et les expérimentations autour de VQGAN et Flow Matching. Pour des détails méthodologiques complémentaires relatifs aux approches de génération/conditionnement de flux, voir également : https://arxiv.org/abs/2211.14575

---

## Architecture et Structure du Projet
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



