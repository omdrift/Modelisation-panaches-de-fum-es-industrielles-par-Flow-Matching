# Smoke Mask Weighting pour Flow Matching

## Problème

Lors de l'entraînement du flow matching, le modèle traite tous les pixels de manière égale, même si les images de fumée contiennent beaucoup de pixels noirs (fond). Cela cause:

- **Reconstruction de fumée partout** : le modèle génère de la fumée même sur le fond noir
- **Perte de détails** : la fumée réelle perd en définition car la loss est diluée sur tous les pixels
- **Gaspillage de capacité** : le modèle apprend à prédire des pixels de fond au lieu de se concentrer sur la fumée

## Solution : Weighted Loss avec Smoke Mask

Le système crée automatiquement un **masque de fumée** et applique une loss pondérée:

### 1. Création du masque ([model/model.py](model/model.py))

```python
# Détecte les pixels avec fumée dans l'espace latent
smoke_threshold = 0.1
smoke_mask = (target_latents.norm(dim=1, keepdim=True) > smoke_threshold).float()
```

**Comment ça marche:**
- Calcule la norme L2 de chaque pixel dans l'espace latent
- Pixels avec norme > 0.1 = fumée (masque = 1)
- Pixels avec norme ≤ 0.1 = fond noir (masque = 0)

### 2. Loss pondérée ([training/trainer.py](training/trainer.py))

```python
# Poids différents pour fumée vs fond
smoke_weight = 5.0      # Fumée 5x plus importante
background_weight = 1.0 # Fond 1x

# Créer la carte de poids
weight_map = smoke_mask * smoke_weight + (1 - smoke_mask) * background_weight

# Loss pondérée
weighted_mse = mse_per_pixel * weight_map
flow_matching_loss = weighted_mse.mean()
```

**Résultat:**
- Pixels de fumée : erreur × 5.0
- Pixels de fond : erreur × 1.0
- Le modèle **se concentre 5x plus sur la fumée**

## Configuration

Dans [configs/smoke_dataset_vqgan.yaml](configs/smoke_dataset_vqgan.yaml):

```yaml
training:
  # Pondération fumée vs fond
  smoke_weight: 5.0        # Poids pour pixels de fumée
  background_weight: 1.0   # Poids pour fond
  smoke_threshold: 0.1     # Seuil de détection en espace latent
```

### Tuning des paramètres

**smoke_weight:**
- `1.0` : pas de pondération (traite tout également)
- `3.0` : focus modéré sur la fumée
- `5.0` : **recommandé** - focus fort sur la fumée
- `10.0` : focus très agressif (peut ignorer complètement le fond)

**smoke_threshold:**
- `0.05` : détecte plus de pixels comme fumée (sensible)
- `0.1` : **recommandé** - équilibré
- `0.2` : détecte seulement la fumée dense (strict)

**background_weight:**
- `0.0` : ignore complètement le fond (risqué)
- `0.5` : fond peu important
- `1.0` : **recommandé** - garde un minimum d'attention sur le fond

## Monitoring

Le système log automatiquement dans wandb:

- `Training/Stats/smoke_ratio` : proportion de pixels détectés comme fumée (0-1)
- `Training/Loss/flow_matching_mse` : loss MSE pondérée

**Valeurs attendues pour smoke_ratio:**
- `0.1-0.3` : fumée occupe 10-30% de l'image (typique)
- `< 0.05` : très peu de fumée (vérifier threshold)
- `> 0.5` : beaucoup de fumée ou threshold trop bas

## Résultats attendus

Avec le masque activé:

 **Plus de détails dans la fumée** : le modèle se concentre sur les structures importantes  
 **Moins de "hallucinations"** : moins de fumée générée sur le fond noir  
 **Meilleure cohérence temporelle** : les mouvements de fumée sont mieux appris  
 **Convergence plus rapide** : le modèle apprend plus efficacement  

## Tests

Pour tester différentes configurations:

```bash
# Configuration aggressive (focus maximal fumée)
python train.py --config configs/smoke_dataset_vqgan.yaml --run-name smoke_aggressive

# Modifier dans le config:
# smoke_weight: 10.0
# smoke_threshold: 0.05

# Configuration conservatrice (équilibrée)
python train.py --config configs/smoke_dataset_vqgan.yaml --run-name smoke_balanced

# Modifier dans le config:
# smoke_weight: 3.0
# smoke_threshold: 0.15
```

## Désactiver le masque

Pour revenir à la loss standard (tous pixels égaux), commentez dans [training/trainer.py](training/trainer.py):

```python
# Apply smoke mask if available (focus loss on smoke regions)
if False:  # Désactivé
    # ... code du masque
```

Ou mettez `smoke_weight: 1.0` et `background_weight: 1.0` dans le config.
