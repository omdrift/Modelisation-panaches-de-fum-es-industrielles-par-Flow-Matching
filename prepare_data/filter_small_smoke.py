import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse


def calculate_smoke_ratio(image_path, threshold=0.1):
    """
    Calculate the ratio of smoke pixels (non-black pixels) in an image.
    
    Args:
        image_path: Path to the image
        threshold: Pixel intensity threshold (0-1) to consider as non-background
    
    Returns:
        Ratio of smoke pixels to total pixels
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Consider a pixel as smoke if any channel > threshold
        smoke_mask = np.max(img_array, axis=2) > threshold
        smoke_ratio = np.sum(smoke_mask) / smoke_mask.size
        
        return smoke_ratio
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return 0.0


def filter_dataset(data_root, split_file, output_file, min_smoke_ratio=0.01, max_smoke_ratio=0.95):
    """
    Filter dataset to remove samples with too small or too large smoke regions.
    
    Args:
        data_root: Root directory containing the dataset
        split_file: Input split file (e.g., train.txt)
        output_file: Output filtered split file
        min_smoke_ratio: Minimum smoke pixel ratio to keep (default: 1%)
        max_smoke_ratio: Maximum smoke pixel ratio to keep (default: 95%)
    """
    data_root = Path(data_root)
    
    # Determine split name from file (train, val, or test)
    split_name = split_file.replace('.txt', '').replace('.backup', '')
    
    # Read original split - format is "frame_path label"
    with open(data_root / split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(lines)} frames from {split_file}")
    
    # Group frames by video sequence (extract video ID from frame name)
    from collections import defaultdict
    video_groups = defaultdict(list)
    for line in lines:
        parts = line.split()
        frame_path = parts[0]
        # Extract video ID (everything before _frame_XXXX.png)
        video_id = frame_path.rsplit('_frame_', 1)[0]
        video_groups[video_id].append(line)
    
    print(f"Found {len(video_groups)} unique video sequences")
    
    filtered_lines = []
    stats = {'too_small': 0, 'too_large': 0, 'kept': 0, 'total': 0}
    
    for video_id, frame_lines in tqdm(video_groups.items(), desc=f"Filtering {split_file}"):
        # Sample a few frames from this video to estimate smoke ratio
        sample_size = min(5, len(frame_lines))
        sample_indices = np.linspace(0, len(frame_lines)-1, sample_size, dtype=int)
        
        ratios = []
        for idx in sample_indices:
            frame_line = frame_lines[idx]
            frame_path = frame_line.split()[0]
            full_path = data_root / split_name / frame_path
            
            if full_path.exists():
                ratio = calculate_smoke_ratio(full_path)
                ratios.append(ratio)
        
        if not ratios:
            continue  # Skip if no valid frames found
        
        avg_ratio = np.mean(ratios)
        stats['total'] += 1
        
        # Filter based on smoke ratio
        if avg_ratio < min_smoke_ratio:
            stats['too_small'] += 1
        elif avg_ratio > max_smoke_ratio:
            stats['too_large'] += 1
        else:
            # Keep all frames from this video
            filtered_lines.extend(frame_lines)
            stats['kept'] += 1
    
    # Save filtered split
    output_path = data_root / output_file
    with open(output_path, 'w') as f:
        for line in filtered_lines:
            f.write(f"{line}\n")
    
    # Print statistics
    print(f"\n=== Filtering Results for {split_file} ===")
    print(f"Total video sequences: {stats['total']}")
    print(f"Too small smoke (<{min_smoke_ratio*100:.1f}%): {stats['too_small']}")
    print(f"Too large smoke (>{max_smoke_ratio*100:.1f}%): {stats['too_large']}")
    print(f"Kept sequences: {stats['kept']} ({stats['kept']/max(1,stats['total'])*100:.1f}%)")
    print(f"Kept frames: {len(filtered_lines)}")
    print(f"Saved to: {output_path}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter dataset by smoke foreground size")
    parser.add_argument("--data-root", type=str, default="final_dataset", 
                        help="Root directory of the dataset")
    parser.add_argument("--min-ratio", type=float, default=0.01,
                        help="Minimum smoke ratio (default: 0.01 = 1%%)")
    parser.add_argument("--max-ratio", type=float, default=0.95,
                        help="Maximum smoke ratio (default: 0.95 = 95%%)")
    parser.add_argument("--backup", action="store_true",
                        help="Backup original split files before filtering")
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    # Backup original files if requested
    if args.backup:
        for split in ['train.txt', 'val.txt', 'test.txt']:
            if (data_root / split).exists():
                backup_path = data_root / f"{split}.backup"
                os.system(f"cp {data_root / split} {backup_path}")
                print(f"Backed up {split} to {backup_path}")
    
    # Filter each split
    for split in ['train.txt', 'val.txt', 'test.txt']:
        if (data_root / split).exists():
            output_file = f"{split}.filtered" if not args.backup else split
            filter_dataset(
                data_root=args.data_root,
                split_file=split if args.backup else split,
                output_file=output_file,
                min_smoke_ratio=args.min_ratio,
                max_smoke_ratio=args.max_ratio
            )
    
    print("Filtering complete!")
    print(f"\nTo use filtered dataset, replace your split files with the .filtered versions")
    print(f"Or run with --backup to directly update the split files (with backup)")


if __name__ == "__main__":
    main()
