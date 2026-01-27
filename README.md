# Modelisation-panaches-de-fum-es-industrielles-par-Flow-MatchingSetup Environment

Prerequisites: Install conda (or mamba) and ensure NVIDIA drivers are installed (nvidia-smi works).

Env file: use the environment file env.yml.

Create the env (recommended):

```
# with conda
conda env create -f env.yml -n smoke

# or faster with mamba
mamba env create -f env.yml -n smoke

conda activate smoke
```

Dataset Overview
The dataset is a snapshot of the smoke labeling tool from February 24, 2020. It contains 12,567 video clips from three different industrial monitoring sites.

Targeted Smoke Labels
For this project, we prioritize the following labels representing confirmed smoke emissions:

47 (Gold Standard Positive): Researcher-confirmed smoke used for data quality checks.

23 (Strong Positive): Two volunteers agreed (or one researcher confirmed) that the video contains smoke.

Getting Started
1. Requirements
Python 3.9+

requests library (for downloading videos)

pandas (for metadata processing)

Bash

pip install requests pandas
2. Download Data
To download the specific smoke videos (Labels 23 and 47), use the provided metadata file metadata_02242020.json and run the download script.

The clips are available in two resolutions:

Standard: 180x180 pixels.

High-Res: 320x320 pixels (obtained by replacing /180/ with /320/ in the URL).

Model Visualization
The models typically use architectures like I3D (Inflated 3D ConvNet) to recognize temporal patterns in smoke movement. Below is an example of how Grad-CAM is used to visualize the areas the model identifies as smoke.

Credits & Citations
If you use this dataset or code in your research, please cite the original Project RISE paper:

Yen-Chia Hsu, Ting-Hao (Kenneth) Huang, Ting-Yao Hu, Paul Dille, Sean Prendi, Ryan Hoffman, Anastasia Tsuhlares, Jessica Pachuta, Randy Sargent, and Illah Nourbakhsh. 2021. Project RISE: Recognizing Industrial Smoke Emissions. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2021). https://arxiv.org/abs/2005.06111

Acknowledgments
We thank the CMU CREATE Lab for providing the deep-smoke-machine repository and the public dataset.

License
Code: BSD 3-clause license

Dataset: Creative Commons Zero (CC0) license


# Code for vqgan and flow matching 

@InProceedings{Davtyan_2023_ICCV,
    author    = {Davtyan, Aram and Sameni, Sepehr and Favaro, Paolo},
    title     = {Efficient Video Prediction via Sparsely Conditioned Flow Matching},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {23263-23274}
}


