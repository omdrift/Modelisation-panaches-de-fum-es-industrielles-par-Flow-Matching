import os
import shutil
import json
import random
from glob import glob
from tqdm import tqdm

def organize_dataset(src_root, dst_root, split_ratio=0.8):
    # 1. Setup Target Directories
    train_path = os.path.join(dst_root, "train")
    test_path = os.path.join(dst_root, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # List all video folders, excluding hidden/internal folders
    video_folders = [f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f)) and not f.startswith('_')]
    random.seed(42) # For reproducibility
    random.shuffle(video_folders) 

    split_idx = int(len(video_folders) * split_ratio)
    train_folders = video_folders[:split_idx]
    test_folders = video_folders[split_idx:]

    stats = {"train": {}, "test": {}}
    
    # We maintain separate lists for the split files
    train_labels = []
    test_labels = []

    def process_split(folders, target_dir, split_name, label_list):
        print(f"Processing split: {split_name}...")
        for video_id in tqdm(folders):
            src_dir = os.path.join(src_root, video_id)
            frames = sorted(glob(os.path.join(src_dir, "*.png")))
            
            stats[split_name][video_id] = len(frames)
            
            for frame_path in frames:
                frame_name = os.path.basename(frame_path)
                # Rename to avoid collisions: videoID_frameName.png
                new_name = f"{video_id}_{frame_name}"
                dst_file = os.path.join(target_dir, new_name)
                
                # Copy the frame
                shutil.copy(frame_path, dst_file)
                
                # Append to specific label list (Format: folder/filename label)
                label_list.append(f"{split_name}/{new_name} 1")

    # 2. Execute Movement and Label Generation
    process_split(train_folders, train_path, "train", train_labels)
    process_split(test_folders, test_path, "test", test_labels)

    # 3. Save Metadata and Split Label Files
    # Save the JSON stats (Important for Flow Matching sequence boundaries)
    with open(os.path.join(dst_root, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Save train_files.txt
    with open(os.path.join(dst_root, "train_files.txt"), "w") as f:
        f.write("\n".join(train_labels))
        
    # Save test_files.txt
    with open(os.path.join(dst_root, "test_files.txt"), "w") as f:
        f.write("\n".join(test_labels))

    with open(os.path.join(dst_root, "labels.txt"), "w") as f:
        f.write("\n".join(train_labels + test_labels))

    print(f"\nOrganization complete!")
    print(f"Stats saved: dataset_stats.json")
    print(f"Labels saved: train_files.txt ({len(train_labels)} lines) and test_files.txt ({len(test_labels)} lines)")

SOURCE_DIR = "isolated_smoke_frames/"
DEST_DIR = "final_dataset/"

organize_dataset(SOURCE_DIR, DEST_DIR)