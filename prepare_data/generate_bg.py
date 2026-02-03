import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_master_backgrounds(base_dir, output_bg_dir):
    os.makedirs(output_bg_dir, exist_ok=True)
    view_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for view_id in tqdm(view_folders, desc="Génération des Master Backgrounds"):
        view_path = os.path.join(base_dir, view_id)
        video_files = [f for f in os.listdir(view_path) if f.endswith('.mp4')]
        
        all_frames = []
        # On pioche 5 frames au hasard par vidéo pour construire le background
        for v_name in video_files[:10]: # Limité à 10 vidéos par vue pour la rapidité
            cap = cv2.VideoCapture(os.path.join(view_path, v_name))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(5):
                cap.set(cv2.CAP_PROP_POS_FRAMES, np.random.randint(0, count))
                ret, frame = cap.read()
                if ret: all_frames.append(frame)
            cap.release()
            
        if all_frames:
            # Calcul de la médiane sur tous les dossiers (supprime la fumée mobile)
            master_bg = np.median(np.stack(all_frames), axis=0).astype(np.uint8)
            cv2.imwrite(os.path.join(output_bg_dir, f"{view_id}_bg.png"), master_bg)

if __name__ == "__main__":
    # Dossier où tu as organisé tes vidéos par view_X-Y
    INPUT_DIR = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos" 
    # Dossier où stocker les images de référence
    BG_DIR = "master_backgrounds"
    generate_master_backgrounds(INPUT_DIR, BG_DIR)