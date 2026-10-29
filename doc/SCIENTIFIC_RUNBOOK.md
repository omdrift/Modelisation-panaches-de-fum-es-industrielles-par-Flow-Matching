# Industrial Plume Forecasting — guide d'exécution

Exécuter les commandes depuis la racine du dépôt dans l'environnement `smoke-science`. Le [protocole](SCIENTIFIC_PROTOCOL.md) fixe la sélection ; le [guide des paramètres et graphiques](TRAINING_DESIGN.md) précise leur interprétation.

## Environnement

```bash
conda env create -f environment_scientific.yml
conda activate smoke-science
python -m scientific.cli doctor
python -m pytest -q
```

Pour un environnement existant :

```bash
conda env update -n smoke-science -f environment_scientific.yml
```

L'étude utilise Optuna 4.6.0. L'installation ciblée est `python -m pip install optuna==4.6.0`. CUDA exige un pilote NVIDIA compatible ; LPIPS charge les poids officiels AlexNet et sa calibration au premier usage. Précharger ces poids pour une exécution hors ligne.

## Étude complète à partir des NPZ

Conserver le dossier préparé entier, notamment `dataset_manifest.json`, les manifests de partitions et `clips/*.npz`. Les MP4 supprimés et le cache CLIPSeg ne sont pas requis pour exploiter une préparation complète. La tâche reste 36 images stockées, 64×64 RGB, cinq observations et quinze futurs.

La prévisualisation ne crée aucun fichier, ne charge pas Optuna et ne démarre aucun entraînement :

```bash
python -m scientific.cli study \
  --config configs/experiment_study.yaml \
  --data scientific_data/v1_2_clipseg_fast \
  --output runs_scientific/plume_study \
  --dry-run
```

Pour lancer ensuite tous les entraînements et évaluations dans l'ordre déclaré :

```bash
python -m wandb login
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m scientific.cli study \
  --config configs/experiment_study.yaml \
  --data scientific_data/v1_2_clipseg_fast \
  --output runs_scientific/plume_study \
  --wandb
```

Sans `--wandb`, les journaux structurés et les sorties locales restent disponibles. Ajouter `--resume` à la même commande pour reprendre une étude. Une reprise exige les mêmes données, sources, configurations et empreintes des artefacts terminés ; changer un réglage nécessite un nouveau dossier d'étude.

La séquence contient au maximum 29 entraînements : six VQGAN exploratoires, six confirmations VQGAN, deux prévisions comparant les VQGAN, six Flow exploratoires, six confirmations Flow et trois Flow pondérés. Les explorations et prévisions comparatives ont un plafond de 15 000 mises à jour ; les confirmations et la variante pondérée de 50 000. Chaque run démarre d'une nouvelle initialisation contrôlée. Les graines de confirmation sont 17, 42 et 2026.

Les candidats croisent LR `5e-5`, `1e-4`, `2e-4` et scheduler constant/cosinus. Le cosinus termine à 10 % du LR initial. Le lot effectif reste 16 ; les micro-lots et l'accumulation adaptent l'occupation mémoire.

Les audits de recherche lisent train et validation seulement. Les scores et images de test restent réservés jusqu'à `final_selection.json`, qui fige les configurations et empreintes retenues. L'audit final et les évaluations des deux partitions de test interviennent ensuite. Une anomalie de partition est rapportée sans déplacement automatique d'images.

### Sorties de l'étude

| Fichier ou dossier | Contenu |
|---|---|
| `study.json` | Identité et progression des étapes |
| `optuna.sqlite3` | Essais, paramètres, résultats intermédiaires et pruning |
| `experiments.csv` | Tableau comparatif des essais et états |
| `report.json`, `report.md` | Rapport automatique et décisions |
| `runs/<identifiant>/protocol.yaml` | Configuration résolue de chaque run |
| `runs/<identifiant>/autoencoder/` ou dossier de prédicteur | Métriques, diagnostics, médias et checkpoints |
| `final_selection.json` | Sélection gelée avant consultation du test |
| `evaluation/` | Métriques, prédictions et médias finaux |

Les maxima ne sont pas des durées en heures. Le coût dépend notamment du GPU, des validations autorégressives et des quatre réalisations des confirmations. Mesurer les temps sur le matériel effectif avant d'annoncer une échéance.

## Validation et arrêt

| Modèle dans l'étude | Intervalle de validation | Patience |
|---|---:|---:|
| VQGAN | 250 mises à jour | 10 validations |
| Flow | 2 500 mises à jour | 4 validations |

Le comptage commence après 5 000 mises à jour et la phase initiale. Une amélioration significative doit dépasser `1e-4` LPIPS. `best.pth` garde indépendamment le minimum brut. Une fin sur plateau est une fin complète ; une interruption ou un échec ne l'est pas.

Les explorations Flow utilisent une réalisation fixe par fenêtre. Les confirmations, prévisions comparant les VQGAN et tests utilisent la moyenne de quatre réalisations, sans retenir la meilleure. Les métriques enregistrent le nombre effectif.

Le pruning médian concerne uniquement l'exploration, après trois essais terminés et 7 500 mises à jour. Les confirmations ne sont pas prunées par comparaison avec d'autres essais. Les essais incomplets, prunés, non finis ou en échec restent visibles dans les rapports et ne sont pas promus comme finalistes terminés.

## Images et monitoring

Les panneaux fixes train et validation utilisent jusqu'à 60 fenêtres chacun. Huit identifiants par partition sont réservés aux illustrations. VQGAN produit les reconstructions des 36 images ; Flow présente les cinq observations et les quinze futurs. PNG et GIF conservent les comparaisons réel/prédiction/erreur. Les illustrations de test sont exportées après gel de la sélection.

Les métriques locales sont dans `metrics.jsonl`. Chaque validation exporte les mesures détaillées, identifiants et couvertures régionales sous `diagnostics/`. Le graphique local de synthèse et les rapports permettent une lecture sans service externe.

W&B sépare VQGAN, Flow uniforme et Flow pondéré. Les courbes montrent qualité train/validation, meilleur LPIPS, patience, codebook VQGAN, objectifs Flow fixes, LR, gradients, ressources et tendances. Les compteurs détaillés restent enregistrés sans générer un graphique pour chaque date, vue ou nombre d'images.

L'objectif Flow mesure une erreur de vitesse dans l'espace latent ; la LPIPS mesure les images décodées et commande la sélection. La perte d'un mini-lot est distincte des objectifs évalués sur panneaux fixes. Le [guide de lecture](TRAINING_DESIGN.md#interprétation-des-graphiques) précise unités et limites.

### Vues W&B

L'éditeur de tableaux utilise un environnement séparé :

```bash
python -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install -r tools/requirements_dashboard.txt
.venv-dashboard/bin/python tools/configure_wandb.py \
  --run-directory runs_scientific/smoke/autoencoder --apply
```

`--run-directory` désigne le dossier d'un run existant contenant `wandb.json` et `run.json`. Pour l'observateur de campagne, il contient `wandb.json` et `status.json`. Omettre `--apply` affiche le plan de présentation, avec lecture du projet W&B en ligne. Une identité hors ligne refuse la configuration des vues distantes avant tout contact réseau.

La vue nommée est filtrée par méthode, avec guide d'interprétation, panneaux de qualité et exemples fixes ; les analyses détaillées sont repliées. L'éditeur conserve les filtres et sélections existants, sauvegarde la vue précédente puis vérifie la persistance. Il ne réécrit ni l'historique ni les poids.

Pour observer les artefacts d'une campagne `pipeline` existante :

```bash
.venv-dashboard/bin/python tools/monitor_campaign.py \
  --campaign runs_scientific/smoke --follow --configure-views
```

Cet observateur lit l'état sans lancer d'entraînement et utilise son propre verrou. Il respecte le mode hors ligne de la campagne ; dans ce cas, omettre `--configure-views`. Un test incomplet reste indiqué comme tel. Les erreurs du suivi ne modifient pas l'état d'entraînement.

## Checkpoints, reprise et fine-tuning

Pendant un entraînement, la rotation conserve deux checkpoints numérotés récents et `best.pth`. Les écritures sont atomiques ; la rotation suit une sauvegarde réussie. Prévoir l'espace temporaire nécessaire à un checkpoint complet. Après clôture de l'étude, les poids sélectionnés et preuves expérimentales restent conservés ; les finalistes gardent leurs états complets. Les médias récents, du meilleur checkpoint et finaux sont retenus sans accumulation de chaque aperçu intermédiaire.

Après la clôture durable de chaque essai, les copies numérotées de récupération sont retirées. Les explorations terminées conservent `weights.pth` avec un reçu de rétention ; les finalistes conservent leur `best.pth` complet. La reprise de l'étude réutilise le registre et les empreintes des essais terminés. Une reprise exacte de boucle concerne les essais interrompus, dont les checkpoints de récupération sont conservés.

Un pilote VQGAN synthétique mesure trois mises à jour adversariales en FP32 avant l'entraînement ; les pilotes prédictifs mesurent ensuite débit et mémoire. Le temps de validation, les animations et les exports s'ajoutent aux estimations issues du seul débit d'optimisation.

Les checkpoints restaurent modèle, optimiseurs, schedulers, RNG et scaler si `training.amp` est activé. La précision mixte est désactivée par défaut. L'arrêt anticipé et les validations en attente sont persistés. La reprise d'un run terminé ne change pas son résultat.

Un fine-tuning est un nouveau run avec un parent identifié. Il ne s'effectue pas en changeant le LR d'une reprise exacte. Dans une configuration dédiée :

```yaml
training:
  init_from:
    path: /chemin/vers/le/checkpoint_parent.pth
    reset: [optimizer, scheduler, rng, early_stopping]
```

Le chargement des poids doit être strictement compatible. Pour un parent VQGAN sans discriminateur, ajouter explicitement `discriminator` aux composants réinitialisés. Les initialisations parentes et réinitialisations sont enregistrées. Cette option ne remplace pas les nouvelles initialisations de la comparaison principale.

## Préparer un nouveau dataset

```bash
python -m scientific.cli inventory --output scientific_data/inventory.json
python -m scientific.cli download --videos smoke_videos
python -m scientific.cli prepare \
  --videos smoke_videos --output scientific_data/smoke \
  --cache-dir scientific_models/huggingface --device cuda --batch-frames 1
```

La préparation utilise CLIPSeg et un raffinement spatial. `--batch-frames` règle le lot de calcul d'images indépendantes ; il ne change pas les 36 acquisitions retenues. `--local-files-only` utilise un cache rempli. Une nouvelle segmentation exige les sources MP4 ; une préparation complète reste exploitable après leur suppression.

La commande `check-data` réalise un contrôle complet, y compris la lecture des clips de test. Pour la recherche contrôlée, laisser `study` limiter d'abord les audits à train/validation ; utiliser le contrôle complet après gel de la sélection.

## Campagne à configuration fixe

La commande `pipeline` orchestre préparation ou réutilisation des données, vérifications, pilote synthétique, VQGAN, références apprises, tests et synthèse. Elle reste distincte de la recherche automatique :

```bash
python -m scientific.cli --context-frames 5 --horizon 15 pipeline \
  --data scientific_data/v1_2_clipseg_fast \
  --campaign runs_scientific/smoke \
  --wandb --wandb-project smoke-science
```

Ajouter `--dry-run` pour la liste des étapes, `--resume` pour reprendre. Lors d'une reprise, sélectionner si nécessaire le protocole figé avec `--protocol runs_scientific/smoke/protocol.yaml` avant le sous-commande `pipeline`. Ses paramètres par défaut, dont les intervalles de validation, sont définis dans `configs/scientific_protocol.yaml` ; l'étude fournit ses propres réglages explicitement.

Les commandes individuelles `pilot`, `autoencoder`, `train`, `inspect-autoencoder`, `evaluate` et `summarize` restent disponibles. `inspect-autoencoder` lit train/validation sans modifier les poids. Les tests réservés ne doivent pas être lancés manuellement pour départager des candidats.

## Vérifications logicielles

```bash
python -m pytest -q
python -m compileall -q scientific model training lutils tools tests
git diff --check
```

Ces vérifications utilisent des fixtures et des exécutions CPU courtes. Elles n'entraînent pas automatiquement la campagne GPU et ne constituent pas des mesures de qualité prédictive sur RISE.
