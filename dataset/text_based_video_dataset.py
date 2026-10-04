# dataset/text_based_video_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.augmentations import build_frame_transform, load_frame
from dataset.sampling import sample_temporal_indices


class TextBasedVideoDataset(Dataset):
    """
    Dataset pour charger des vidéos à partir d'un fichier texte listant les noms de frames
    """

    def __init__(
            self,
            data_path: str,
            file_list: str,  # train_files.txt ou val_files.txt
            input_size: int,
            crop_size: int,
            frames_per_sample: int = 18,
            random_horizontal_flip: bool = True,
            random_time: bool = True,
            skip_short_videos: bool = False,
            skip_frames: int = 1):
        
        self.data_path = data_path
        self.input_size = input_size
        self.crop_size = crop_size
        self.frames_per_sample = frames_per_sample
        self.random_horizontal_flip = random_horizontal_flip
        self.random_time = random_time
        self.skip_short_videos = skip_short_videos
        self.skip_frames = skip_frames
        
        # Charger la liste des séquences depuis le fichier texte
        file_list_path = os.path.join(data_path, file_list)
        with open(file_list_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # accept formats: "filename", "filename 1", or "train/filename 1"
        self.sequence_names = [os.path.basename(l.split()[0]) for l in lines]
        if self.frames_per_sample > 1:
            # Split files list every frame. Flow Matching samples one clip per video.
            first_name_by_video = {}
            for sequence_name in self.sequence_names:
                first_name_by_video.setdefault(self.get_video_name(sequence_name), sequence_name)
            self.sequence_names = list(first_name_by_video.values())
        
        # Dossier contenant les images - detect train/val/test subdirectories
        if 'train' in file_list:
            self.images_dir = os.path.join(data_path, 'train')
        elif 'val' in file_list:
            self.images_dir = os.path.join(data_path, 'val')
        elif 'test' in file_list:
            self.images_dir = os.path.join(data_path, 'test')
        else:
            self.images_dir = os.path.join(data_path, 'images')        
        # Construire un mapping des vidéos et leurs frames disponibles
        print(f"Scanning available frames...")
        self.video_frames = self._scan_available_frames()

        if self.frames_per_sample > 1:
            required = 1 + (self.frames_per_sample - 1) * self.skip_frames
            short_video_ids = {
                video_name
                for video_name, info in self.video_frames.items()
                if len(info.get("indices", [])) < required
            }
            if short_video_ids and self.skip_short_videos:
                before = len(self.sequence_names)
                self.sequence_names = [
                    name for name in self.sequence_names
                    if self.get_video_name(name) not in short_video_ids
                ]
                print(
                    f"Skipped {len(short_video_ids)} videos shorter than {required} frames "
                    f"({before - len(self.sequence_names)} clip entries)."
                )
            if not self.sequence_names:
                raise ValueError(
                    f"No videos in {file_list} have the {required} frames needed "
                    f"for a {self.frames_per_sample}-frame sample"
                )
        
        # Transformations
        self.transform = build_frame_transform(self.input_size, self.crop_size)
        
        print(f"Dataset loaded: {len(self.sequence_names)} sequences from {len(self.video_frames)} videos")

    def _scan_available_frames(self):
        """Scanne le dossier images pour trouver toutes les frames disponibles par vidéo"""
        video_frames = {}

        # Parcourir les noms de séquences pour identifier les vidéos uniques
        unique_videos = set(self.get_video_name(s) for s in self.sequence_names)

        # Fast single-pass: lister once images_dir and group indices per video
        try:
            all_files = os.listdir(self.images_dir)
        except FileNotFoundError:
            return {v: {"indices": [], "max_idx": 0} for v in unique_videos}

        temp_map = {}
        for fname in all_files:
            if not fname.endswith('.png'):
                continue
            if '_frame_' not in fname:
                continue
            video_name, tail = fname.rsplit('_frame_', 1)
            # only keep videos we care about to save memory
            if video_name not in unique_videos:
                continue
            idx_str = tail.replace('.png', '')
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            temp_map.setdefault(video_name, []).append(idx)

        # Build final mapping
        for video_name in unique_videos:
            indices = temp_map.get(video_name, [])
            indices = sorted(set(indices))
            max_idx = indices[-1] if indices else 0
            video_frames[video_name] = {"indices": indices, "max_idx": max_idx}

        return video_frames

    def __len__(self):
        return len(self.sequence_names)
    
    def get_video_name(self, sequence_name):
        """Extrait le nom de la vidéo depuis le nom de la séquence"""
        # Trouver la dernière occurrence de '_frame_' et tout retirer après
        if '_frame_' in sequence_name:
            return sequence_name[:sequence_name.rfind('_frame_')]
        else:
            return sequence_name.rsplit('_', 1)[0]
    
    def get_frame_number(self, sequence_name):
        """Extrait le numéro de frame depuis le nom de la séquence"""
        # Exemple: clairton1_2018-12-13_frame9506_24_f0014 -> 14
        frame_str = sequence_name.split('_')[-1]
        return int(frame_str.replace('f', ''))

    def __getitem__(self, index):
        sequence_name = self.sequence_names[index]
        video_name = self.get_video_name(sequence_name)
        
        # Determine the available indices for this video.
        vf_info = self.video_frames.get(video_name, {"indices": [], "max_idx": 0})
        available_indices = vf_info.get("indices", [])
        if not available_indices:
            raise ValueError(f"No readable frames found for video '{video_name}'")

        # Apply one shared augmentation decision to every frame in the clip.
        flip_p = np.random.rand() < 0.5 if self.random_horizontal_flip else False
        sampled_indices = sample_temporal_indices(
            available_indices,
            self.frames_per_sample,
            random_time=self.random_time,
            stride=self.skip_frames,
        )

        frames = []
        for frame_idx in sampled_indices:
            frame_path = os.path.join(self.images_dir, f"{video_name}_frame_{frame_idx:04d}.png")
            if not os.path.isfile(frame_path):
                raise FileNotFoundError(f"Frame listed for video '{video_name}' is missing: {frame_path}")
            frames.append(load_frame(frame_path, self.transform, horizontal_flip=flip_p))

        return torch.stack(frames, dim=0)
