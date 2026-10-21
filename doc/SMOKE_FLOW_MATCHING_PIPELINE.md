# Pipeline Flow Matching actuel

Les commandes complètes et les prérequis sont maintenus dans le [README](../README.md).
Les données et checkpoints ne sont pas inclus dans un clone du dépôt.

## Étapes et artefacts

| Étape | Entrée | Sortie |
|---|---|---|
| Téléchargement | `prepare_data/metadata_02242020.json`, URL accessibles | MP4 dans `smoke_videos/` |
| Extraction et matting | MP4 locaux | `frames/<video>/frame_0000.png` |
| Organisation | Frames extraites | `final_dataset/{train,val,test}/` et `{train,val,test}.txt` |
| VQGAN | `configs/vqgan_64.yaml`, images 64×64 | `runs_vqgan/vqgan_64_baseline/checkpoints/vqgan_epoch_50.ckpt` après le run de 50 epochs |
| Flow Matching | `configs/flow_matching_64.yaml`, poids VQGAN | `runs/flow_smoke_v1/checkpoints/step_100000.pth` après le run de 100000 steps |
| Évaluation | Configuration du run et ses deux checkpoints | Métriques, grilles, GIF et frames dans `evaluation_outputs/` |

Ces chemins indiquent ce que les commandes du README produisent ; ils ne
désignent pas des fichiers déjà disponibles. Le script d'organisation refuse
de remplacer une destination non vide. Les splits sont séparés par vidéo source.

## Objectif d'entraînement

Le preset principal encode des images 64×64 RGB en latents 8×8×256. L'autoencodeur
est figé pendant le train FM. Pour chaque fenêtre de 10 observations, le sampler
choisit une cible `tau`, sa référence `tau - 1` et une condition `c < tau - 1`.
La distance utilisée par le régresseur est `(tau - 1) - c`.
Le Flow Matching interpole entre un bruit gaussien et le latent cible, puis
régresse le champ vectoriel associé. La loss peut pondérer les régions détectées
comme fumée dans le latent ; ce masque n'est pas une annotation de segmentation.

`condition_frames` configure l'inférence, pas le nombre de conditions du sampler
d'entraînement. Le preset principal génère 6 frames à partir d'une frame initiale.
La fenêtre de validation charge 16 frames (10 observations + 6 futures), tandis que
la génération et l'évaluation autonome comparent les 6 futures immédiatement
après la condition initiale. Les clips trop courts sont exclus, sans répétition
artificielle de leur dernière frame.

## Paramètres et sorties

- `training.eval_every` et `training.checkpoint_every` sont exprimés en steps.
- `evaluation.steps` est appliqué par l'évaluateur du train et l'évaluateur autonome.
- Le train FM conserve une copie du YAML au premier lancement dans `runs/<run-name>/config.yaml`.
- L'évaluation autonome calcule MSE, PSNR et, si disponible, SSIM image. Le FVD et le SSIM temporel ne sont pas implémentés dans ce parcours.
- Le nombre de frames d'une liste n'est pas le nombre de clips du dataset FM : les entrées sont regroupées par vidéo source.

## Variantes et résultats historiques

Le preset 128 px utilise des latents 16×16×256 et exige un autoencodeur compatible.
Les fichiers `smoke_dataset*.yaml` sont des variantes ; leurs budgets de steps et
horizons peuvent différer du preset principal. Ils ne permettent pas de déduire
la configuration des médias historiques. Consulter le [registre](EXPERIMENT_REGISTRY.md)
avant d'associer un résultat à un checkpoint.
