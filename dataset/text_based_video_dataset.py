# dataset/text_based_video_dataset.py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


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
            random_time: bool = True):
        
        self.data_path = data_path
        self.input_size = input_size
        self.crop_size = crop_size
        self.frames_per_sample = frames_per_sample
        self.random_horizontal_flip = random_horizontal_flip
        self.random_time = random_time
        
        # Charger la liste des séquences depuis le fichier texte
        file_list_path = os.path.join(data_path, file_list)
        with open(file_list_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        # accept formats: "filename", "filename 1", or "train/filename 1"
        self.sequence_names = [os.path.basename(l.split()[0]) for l in lines]
        
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
        
        # Transformations
        self.transform = T.Compose([
            T.Resize(size=self.input_size, antialias=True),
            T.CenterCrop(size=self.crop_size),
        ])
        
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
        
        # Déterminer les indices disponibles pour cette vidéo
        vf_info = self.video_frames.get(video_name, {"indices": [], "max_idx": 0})
        available_indices = vf_info.get("indices", [])
        max_idx = vf_info.get("max_idx", 0)
        total_frames = max_idx + 1 if max_idx >= 0 else 18
        
        # Si random_time, choisir un offset aléatoire
        if self.random_time:
            max_offset = max(0, total_frames - self.frames_per_sample)
            time_offset = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
        else:
            time_offset = 0
        
        # Déterminer si on fait un flip horizontal
        flip_p = np.random.rand() < 0.5 if self.random_horizontal_flip else False
        
        # Charger les frames
        frames = []
        # Helper: find nearest existing frame index using binary search
        import bisect
        def find_nearest(sorted_list, target):
            if not sorted_list:
                return None
            pos = bisect.bisect_left(sorted_list, target)
            if pos == 0:
                return sorted_list[0]
            if pos == len(sorted_list):
                return sorted_list[-1]
            before = sorted_list[pos - 1]
            after = sorted_list[pos]
            if abs(before - target) <= abs(after - target):
                return before
            else:
                return after

        for frame_idx in range(self.frames_per_sample):
            desired_num = time_offset + frame_idx
            if desired_num > max_idx:
                desired_num = max_idx

            nearest = find_nearest(available_indices, desired_num)
            if nearest is None:
                # no frames available for this video -> black frame
                print(f"Warning: No frames found for video: {video_name}")
                img = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
            else:
                frame_path = os.path.join(self.images_dir, f"{video_name}_frame_{nearest:04d}.png")
                if not os.path.exists(frame_path):
                    print(f"Warning: Frame not found (after nearest search): {frame_path}")
                    img = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
                else:
                    img = Image.open(frame_path).convert("RGB")
            
            # Appliquer les transformations
            img_tensor = T.functional.to_tensor(img)
            img_tensor = self.transform(img_tensor)
            
            # Flip horizontal si nécessaire
            if flip_p:
                img_tensor = T.functional.hflip(img_tensor)
            
            frames.append(img_tensor)
        
        # Empiler toutes les frames
        video = torch.stack(frames, dim=0)
        
        # Normaliser de [0, 1] à [-1, 1]
        video = video * 2.0 - 1.0
        
        return video