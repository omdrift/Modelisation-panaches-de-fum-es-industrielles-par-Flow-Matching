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
            self.sequence_names = [line.strip() for line in f.readlines()]
        
        # Dossier contenant les images
        # Dossier contenant les images - detect train/ or test/ subdirectories
        if 'train' in file_list:
            self.images_dir = os.path.join(data_path, 'train')
        elif 'test' in file_list or 'val' in file_list:
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
        unique_videos = set()
        for seq_name in self.sequence_names:
            video_name = self.get_video_name(seq_name)
            unique_videos.add(video_name)
        
        # Pour chaque vidéo unique, compter les frames disponibles
        for video_name in unique_videos:
            frame_count = 0
            for i in range(50):  # Chercher jusqu'à 50 frames max
                frame_path = os.path.join(self.images_dir, f"{video_name}_frame_{i:04d}.png")
                if os.path.exists(frame_path):
                    frame_count = i + 1
                else:
                    break
            video_frames[video_name] = max(frame_count, 18)  # Au moins 18 frames par défaut
        
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
        
        # Déterminer le nombre de frames disponibles pour cette vidéo
        total_frames = self.video_frames.get(video_name, 18)
        
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
        for frame_idx in range(self.frames_per_sample):
            frame_num = time_offset + frame_idx
            if frame_num >= total_frames:
                frame_num = total_frames - 1  # Utiliser la dernière frame si on dépasse
            
            frame_path = os.path.join(
                self.images_dir, 
                f"{video_name}_frame_{frame_num:04d}.png"
            )
            
            # Charger l'image
            try:
                img = Image.open(frame_path).convert("RGB")
            except FileNotFoundError:
                # Si la frame n'existe pas, créer une frame noire
                print(f"Warning: Frame not found: {frame_path}")
                img = Image.new('RGB', (self.crop_size, self.crop_size), (0, 0, 0))
            
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