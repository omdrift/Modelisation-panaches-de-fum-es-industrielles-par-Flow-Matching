# Industrial Plume Forecasting — architecture

Le dépôt expose deux parcours : `study` compare des configurations selon un plan versionné ; `pipeline` exécute une recette fixe avec les références. Les commandes de préparation, d'inspection et d'évaluation restent disponibles séparément. Les données préparées sont réutilisables sans les vidéos sources.

## Données et absence de fuite temporelle

```mermaid
flowchart TD
    A[RISE: métadonnées et MP4] --> B[Décodage des 36 acquisitions]
    B --> C[CLIPSeg indépendant par image et filtre spatial]
    C --> D[NPZ: raw_frames, frames, alpha en 64x64]
    D --> E[Manifests: identifiants, dates, vues, temps et empreintes]
    E --> F[Train et validation]
    E --> G[Tests réservés par dates et vues]
    F --> H[VQGAN et prédicteur]
    I[Cinq observations transformées] --> H
    H --> J[Quinze images futures générées]
    G --> K[Comparaison après gel de la sélection]
    J --> K
```

La segmentation d'une observation utilise seulement cette image et des poids/paramètres figés. Une autre image du clip ne doit modifier ni son RGB transformé ni son masque. Le générateur reçoit cinq observations et réutilise ensuite ses propres prédictions ; les vraies images futures et leurs masques servent uniquement à la mesure.

L'apprentissage Flow sélectionne une cible, sa référence précédente et une condition antérieure parmi les 36 images. Ce conditionnement temporel parcimonieux se distingue d'une perte directe sur quinze images futures. Le temps de transport de l'ODE se distingue des horodatages des acquisitions.

Les audits de recherche lisent train et validation. Les manifestes identifient les partitions réservées ; les pixels et résultats du test sont différés jusqu'au gel des choix. Les doublons exacts sont recherchés sur les RGB avant segmentation ; une forte similarité est signalée sans changement automatique de partition.

## Sélection progressive

```mermaid
flowchart TD
    A[Configuration d'étude et identité des données] --> B[Six essais VQGAN: graine 42]
    B --> C[Deux finalistes sur trois graines]
    C --> D[Deux prévisions identiques avec VQGAN de graine 42]
    D --> E[Représentation commune sélectionnée et gelée]
    E --> F[Six essais Flow uniforme: graine 42]
    F --> G[Deux finalistes sur trois graines]
    G --> H[Configuration Flow choisie par LPIPS moyen]
    H --> I[Flow pondéré: mêmes réglages et trois graines]
    I --> J[final_selection.json: choix et empreintes]
    J --> K[Audit final et tests dates/vues]
    K --> L[Mesures, médias et rapport]
```

Les phases comprennent au maximum douze entraînements VQGAN et dix-sept Flow. Les candidats croisent trois learning rates et deux schedulers. Les explorations et prévisions comparant les VQGAN ont un plafond de 15 000 mises à jour ; les confirmations et Flow pondérés de 50 000. Les graines de confirmation sont 17, 42 et 2026.

Optuna stocke les essais séquentiels dans SQLite. Son pruning médian compare seulement les explorations, après trois essais terminés et 7 500 mises à jour. Les confirmations sont indépendantes de ce pruning. Le choix de VQGAN utilise la prévision en validation, avec reconstruction moyenne comme départage. Le choix Flow utilise la moyenne des graines en validation. Les tests ne participent à aucune de ces décisions.

## Mesure et contrôle de l'entraînement

```mermaid
flowchart LR
    A[Poids courants] --> B[Évaluation sur panels et tirages fixes]
    T[Train: 60 fenêtres] --> B
    V[Validation: 60 fenêtres] --> B
    B --> C[LPIPS et erreurs RGB/régionales]
    B --> D[Codebook VQGAN ou objectif latent Flow]
    B --> E[Huit clips visuels: PNG et GIF]
    C --> F[Minimum brut: best.pth]
    C --> G[Progrès significatif: patience]
    G --> H[Fin propre sur plateau]
    C --> I[Courbes locales et W&B]
    D --> I
    E --> I
```

Les diagnostics restaurent les RNG et les modes des modules. Flow compare aussi l'objectif latent sur des tirages fixes indépendants ; la sélection reste fondée sur les images générées et décodées. Une réalisation est utilisée en exploration Flow, quatre en confirmation, pour les prévisions comparant les VQGAN et au test. Les réalisations sont toutes incluses dans la moyenne.

Dans l'étude, VQGAN valide toutes les 250 mises à jour avec patience 10 ; Flow toutes les 2 500 avec patience quatre. Le comptage débute à 5 000 mises à jour, avec `min_delta=1e-4`. Le minimum brut sélectionnant `best.pth` est distinct de l'amélioration significative qui remet la patience à zéro. L'arrêt anticipé ne prouve ni convergence globale ni absence de surapprentissage.

## États et reprise

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> running: lancement contrôlé
    running --> complete: sorties et décision valides
    running --> pruned: comparaison intermédiaire autorisée
    running --> failed: erreur ou valeurs non finies
    running --> interrupted: arrêt du processus
    interrupted --> running: reprise compatible
    failed --> running: correction et reprise compatible
    complete --> complete: réutilisation après vérification
```

Une fin par plafond ou arrêt anticipé est complète, avec motif explicite. Une interruption ne constitue pas une sélection finale. Les états Optuna et les états des runs sont reliés dans le registre sans confondre essai pruné et candidat confirmé.

La reprise exige les mêmes identités de code, données, paramètres et artefacts. Elle restaure optimiseurs, schedulers, RNG, patience et scaler AMP si activé. Un fine-tuning crée un autre run avec parent et réinitialisations déclarés.

## Artefacts et présentation

| Niveau | Artefacts |
|---|---|
| Étude | `study.json`, `optuna.sqlite3`, `experiments.csv`, `report.json`, `report.md` |
| Run | Protocole résolu, `effective_training.json`, métriques, diagnostics, médias, checkpoints |
| Décision | `final_selection.json` avec empreintes gelées avant test |
| Évaluation | Métriques par fenêtre/horizon/réalisation, durées, NFE, prédictions NPZ, médias |
| Campagne fixe | `pipeline.json`, protocole figé, logs et synthèse |
| Observateur | `campaign_monitor/status.json`, identité W&B, état de lecture et snapshot |

Le suivi W&B est optionnel ; les sorties locales restent disponibles. L'observateur de campagne lit les artefacts avec un verrou séparé, sans charger les tenseurs ni exécuter l'entraînement. Les vues sont séparées par méthode et accompagnées d'une définition des mesures. Les archives de test présentes indiquent une progression d'écriture, sans constituer un score final.

Pendant l'apprentissage, conserver deux checkpoints de récupération et le meilleur. Les médias récents, du meilleur checkpoint et finaux limitent l'accumulation. Après clôture, les finalistes conservent leurs états complets ; les autres essais gardent leurs poids sélectionnés et preuves expérimentales.

## Publication

Le dépôt contient code, configurations, protocole, références, tests et licences. Vidéos, NPZ, poids, caches, sessions W&B et résultats locaux restent exclus de Git. Une publication de résultats doit relier métriques, configurations et empreintes des artefacts, sans assimiler tests logiciels ou reconstruction d'images à une démonstration de qualité prédictive.
