import argparse
import os
import shutil
import json
import random
from glob import glob
from tqdm import tqdm

def organize_dataset(src_root, dst_root, train_ratio=0.8, val_ratio=0.1):
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")
    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"Frame directory not found: {src_root}")
    if os.path.exists(dst_root) and os.listdir(dst_root):
        raise FileExistsError(f"Destination is not empty: {dst_root}. Use a new directory to preserve existing splits.")
    video_folders = sorted(
        name for name in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, name))
        and not name.startswith(('_', '.'))
        and glob(os.path.join(src_root, name, '*.png'))
    )
    random.Random(42).shuffle(video_folders)
    split_idx1 = int(len(video_folders) * train_ratio)
    split_idx2 = split_idx1 + int(len(video_folders) * val_ratio)
    if not (0 < split_idx1 < split_idx2 < len(video_folders)):
        raise ValueError("Not enough non-empty video directories to create three non-empty splits")
    # 1. Setup Target Directories
    train_path = os.path.join(dst_root, "train")
    test_path = os.path.join(dst_root, "test")
    val_path = os.path.join(dst_root, "val")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

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

def main():
    parser = argparse.ArgumentParser(description="Create frame lists and train/val/test splits by source video.")
    parser.add_argument("--src-root", default="frames")
    parser.add_argument("--dst-root", default="final_dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    organize_dataset(args.src_root, args.dst_root, args.train_ratio, args.val_ratio)


if __name__ == "__main__":
    main()
