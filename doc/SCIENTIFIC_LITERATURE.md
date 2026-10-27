# Industrial Plume Forecasting — références et portée

Cette sélection de sources primaires documente les composants de l'étude. Elle ne constitue pas une revue systématique exhaustive. Les références BibTeX sont dans [references.bib](references.bib).

| Travail | Source primaire | Utilisation et limite |
|---|---|---|
| Hsu et al., Project RISE, AAAI 2021 | [Article](https://arxiv.org/abs/2005.06111), [dataset et partitions](https://github.com/CMU-CREATE-Lab/deep-smoke-machine#dataset) | Corpus de reconnaissance réutilisé pour la prédiction visuelle ; absence de masques denses de référence. |
| Lipman et al., Flow Matching, ICLR 2023 | [Article](https://arxiv.org/abs/2210.02747) | Construction du champ vectoriel et de la cible de transport. |
| Davtyan, Sameni et Favaro, RIVER, ICCV 2023 | [Article](https://arxiv.org/abs/2211.14575), [code](https://github.com/araachie/river) | Conditionnement temporel parcimonieux et génération latente ; les scores publiés ne sont pas des résultats RISE. |
| Esser, Rombach et Ommer, VQGAN, CVPR 2021 | [Article](https://arxiv.org/abs/2012.09841), [code](https://github.com/CompVis/taming-transformers) | Autoencodeur quantifié et apprentissage adversarial ; architecture propre à cette configuration. |
| Gao et al., SimVP, CVPR 2022 | [Article](https://arxiv.org/abs/2206.05099), [code](https://github.com/A4Bio/SimVP) | Référence CNN adaptée au contexte et à l'horizon choisis ; pas de reproduction revendiquée des scores originaux. |
| Zhang et al., LPIPS, CVPR 2018 | [Article](https://arxiv.org/abs/1801.03924), [implémentation](https://github.com/richzhang/PerceptualSimilarity) | Critère perceptuel calibré ; pas d'étalonnage spécifique à la fumée transparente. |
| Wang et al., SSIM, IEEE TIP 2004 | [Page des auteurs](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html) | Mesure structurelle complémentaire au LPIPS. |
| Farnebäck, SCIA 2003 | [Article éditeur](https://link.springer.com/chapter/10.1007/3-540-45103-X_50) | Flot dense pour une extrapolation à mouvement constant. |
| Unterthiner et al., FVD, prépublication | [Article](https://arxiv.org/abs/1812.01717) | Distance distributionnelle vidéo réservée à une extension dimensionnée. |
| Lüddecke et Ecker, CLIPSeg, CVPR 2022 | [Article](https://arxiv.org/abs/2112.10003), [API 4.49](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/clipseg) | Segmentation conditionnée par texte ; scores sémantiques et non opacité physique. |
| He, Sun et Tang, Guided Image Filtering, ECCV 2010 | [Article des auteurs](https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf) | Raffinement spatial par luminance ; aucune garantie de cohérence temporelle. |
| Cawley et Talbot, JMLR 2010 | [Article](https://jmlr.csail.mit.edu/beta/papers/v11/cawley10a.html) | Surajustement du critère de sélection sur une validation finie ; justifie la séparation entre réglages et évaluation finale. |

Avant de rédiger l'article, compléter la recherche sur la prédiction vidéo de fumée, le Flow Matching latent et l'évaluation des prévisions stochastiques. Consigner les requêtes, critères d'inclusion et sources retenues. Distinguer détection, génération conditionnelle, simulation physique et prédiction visuelle. Toute revendication de nouveauté doit être étayée par cette revue et par les expériences.
