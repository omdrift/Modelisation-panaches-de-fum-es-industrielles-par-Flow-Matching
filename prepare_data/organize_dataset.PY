import os
import shutil
import json
import random
from glob import glob
from tqdm import tqdm

def organize_dataset(src_root, dst_root, train_ratio=0.8, val_ratio=0.1):
    # 1. Setup Target Directories
    train_path = os.path.join(dst_root, "train")
    test_path = os.path.join(dst_root, "test")
    val_path = os.path.join(dst_root, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # List all video folders, excluding hidden/internal folders
    video_folders = [f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f)) and not f.startswith('_')]
    random.seed(42) # For reproducibility
    random.shuffle(video_folders) 

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    split_idx1 = int(len(video_folders) * train_ratio)
    split_idx2 = split_idx1 + int(len(video_folders) * val_ratio)

    train_folders = video_folders[:split_idx1]
    val_folders = video_folders[split_idx1:split_idx2]
    test_folders = video_folders[split_idx2:]

    stats = {"train": {}, "val": {}, "test": {}}
    
    train_labels = []
    val_labels = []
    test_labels = []

    def process_split(folders, target_dir, split_name, label_list):
        print(f"Processing split: {split_name}...")
        for video_id in tqdm(folders):
            src_dir = os.path.join(src_root, video_id)
            frames = sorted(glob(os.path.join(src_dir, "*.png")))
            
            stats[split_name][video_id] = len(frames)
            
            for frame_path in frames:
                frame_name = os.path.basename(frame_path)
                new_name = f"{video_id}_{frame_name}"
                dst_file = os.path.join(target_dir, new_name)
                
                shutil.copy(frame_path, dst_file)
                
                label_list.append(f"{split_name}/{new_name} 1")

    process_split(train_folders, train_path, "train", train_labels)
    process_split(val_folders, val_path, "val", val_labels)
    process_split(test_folders, test_path, "test", test_labels)


    with open(os.path.join(dst_root, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Save train.txt, val.txt, test.txt
    with open(os.path.join(dst_root, "train.txt"), "w") as f:
        f.write("\n".join(train_labels))

    with open(os.path.join(dst_root, "val.txt"), "w") as f:
        f.write("\n".join(val_labels))

    with open(os.path.join(dst_root, "test.txt"), "w") as f:
        f.write("\n".join(test_labels))


    print(f"Labels saved: train.txt ({len(train_labels)} lines), val.txt ({len(val_labels)} lines) and test.txt ({len(test_labels)} lines)")

SOURCE_DIR = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_frames/"
DEST_DIR = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/final_dataset/"

organize_dataset(SOURCE_DIR, DEST_DIR)