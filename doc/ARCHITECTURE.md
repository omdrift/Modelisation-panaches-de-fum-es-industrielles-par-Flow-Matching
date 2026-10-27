# Industrial Plume Forecasting — architecture

Ce document décrit les relations entre les données, les modèles et les artefacts. La commande `pipeline` est le point d'entrée canonique ; les commandes individuelles servent au diagnostic ou à la reprise ciblée.

## Vue d'ensemble

```mermaid
flowchart TD
    A[RISE metadata + MP4] --> B[prepare]
    B --> C[36 frames RGB + alpha 64x64]
    C --> D[check-data]
    D --> E[pilot]
    E --> F[VQGAN commun]
    F --> G[SimVP]
    F --> H[Flow Matching uniforme]
    F --> I[Flow Matching pondéré alpha]
    G --> J[Evaluation: test_dates + test_views]
    H --> J
    I --> J
    C --> J
    J --> K[summarize]
    K --> L[summary.json + CSV + graphiques]
```

Chaque flèche représente une dépendance vérifiée par l'orchestrateur. Un stage ne peut pas être marqué terminé si son manifeste, son checkpoint ou ses métriques sont absents.

## Une fenêtre et l'absence de fuite

```mermaid
flowchart LR
    X[Clip complet: 36 images] --> R[RGB source]
    R --> S[CLIPSeg image par image]
    S --> M[alpha soft mask]
    S --> T[RGB de fumée isolée]
    T --> P[observations C=5]
    T --> Y[cibles futures H=15]
    P --> N[prédicteur]
    N --> Q[15 images générées]
    Y --> E[mesures uniquement]
    Q --> E
    M --> E
```

Pendant la génération, le prédicteur reçoit uniquement les cinq observations. Les quinze images futures et leurs masques sont utilisés après la génération pour la perte de validation ou les métriques de test ; ils ne sont jamais transmis à l'entrée du prédicteur.

## Monitoring et sélection

```mermaid
flowchart TD
    A[Poids à une étape de validation] --> B[Inférence sans gradients]
    T[Panel train fixe] --> B
    V[Panel validation fixe] --> B
    B --> C[LPIPS train et validation]
    B --> D[Erreurs fumée et fond + images]
    B --> E[Utilisation du codebook VQGAN]
    C --> F[Tendances et écart de généralisation]
    V --> G[LPIPS de validation pour sélection]
    A --> G
    G --> H[best.pth si amélioration]
    C --> J[WandB + diagnostics JSON]
    D --> J
    E --> J
    F --> J
```

Les diagnostics ne modifient ni l'objectif d'entraînement ni le critère de sélection. Les états aléatoires et les modes des modules sont restaurés après l'inférence. Un écart train/validation ou une faible utilisation du codebook constitue un signal à examiner ; aucun de ces indicateurs ne prouve à lui seul un surapprentissage. Les ensembles de test restent réservés à l'évaluation finale.

## États de campagne

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> running: lancement du stage
    running --> complete: sorties + empreintes valides
    running --> failed: erreur ou sortie incomplète
    running --> interrupted: arrêt utilisateur
    interrupted --> running: pipeline --resume
    failed --> running: correction puis --resume
    complete --> complete: réutilisation après vérification
```

`pipeline.json` conserve l'état de chaque stage, la commande exécutée, le protocole, les chemins d'entrée et les empreintes des sorties. `protocol.yaml` est une copie figée de la configuration utilisée pour la campagne. Les logs sont dans `runs_scientific/<campagne>/logs/`.

## Dossiers publiables

Le dépôt contient le code, le protocole, la bibliographie, les tests et les notices de licence. Les vidéos, les données préparées, les poids CLIPSeg/LPIPS, les checkpoints, les sessions WandB et les sorties de campagne sont locales et ignorées par Git. Publier les métriques et les manifests d'une campagne séparément, après vérification de leur empreinte et de leur licence.
