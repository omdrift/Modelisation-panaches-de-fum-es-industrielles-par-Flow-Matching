import os
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import defaultdict


def calculate_smoke_ratio(image_path, threshold=0.1):
    """Calculate the ratio of smoke pixels in an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        smoke_mask = np.max(img_array, axis=2) > threshold
        smoke_ratio = np.sum(smoke_mask) / smoke_mask.size
        return smoke_ratio
    except Exception as e:
        return 0.0


def visualize_smoke_thresholds(data_root, split='train', num_samples=500):
    """
    Visualize samples at different smoke ratio thresholds.
    """
    data_root = Path(data_root)
    split_file = data_root / f"{split}.txt"
    
    # Read split file
    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Analyzing {len(lines)} frames from {split}.txt")
    
    # Group frames by video
    video_groups = defaultdict(list)
    for line in lines:
        frame_path = line.split()[0]
        video_id = frame_path.rsplit('_frame_', 1)[0]
        video_groups[video_id].append(frame_path)
    
    print(f"Found {len(video_groups)} video sequences")
    
    # Sample videos and calculate smoke ratios
    sample_size = min(num_samples, len(video_groups))
    sampled_videos = list(video_groups.keys())[:sample_size]
    
    video_ratios = []
    print("\nCalculating smoke ratios...")
    
    for video_id in tqdm(sampled_videos):
        frames = video_groups[video_id]
        # Sample first frame from video
        frame_path = frames[0]
        full_path = data_root / split / frame_path
        
        if full_path.exists():
            ratio = calculate_smoke_ratio(full_path)
            video_ratios.append((video_id, ratio, full_path))
    
    # Sort by ratio
    video_ratios.sort(key=lambda x: x[1])
    
    # Create histogram
    ratios = [r[1] for r in video_ratios]
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(ratios, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Smoke Ratio')
    plt.ylabel('Number of Videos')
    plt.title(f'Distribution of Smoke Ratios ({len(ratios)} videos)')
    plt.axvline(x=0.01, color='r', linestyle='--', label='1% threshold')
    plt.axvline(x=0.02, color='orange', linestyle='--', label='2% threshold')
    plt.axvline(x=0.05, color='yellow', linestyle='--', label='5% threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(ratios, bins=50, edgecolor='black', alpha=0.7, cumulative=True, density=True)
    plt.xlabel('Smoke Ratio')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution')
    plt.axvline(x=0.01, color='r', linestyle='--', label='1% threshold')
    plt.axvline(x=0.02, color='orange', linestyle='--', label='2% threshold')
    plt.axvline(x=0.05, color='yellow', linestyle='--', label='5% threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'smoke_ratio_distribution_{split}.png', dpi=150)
    print(f"\nSaved histogram to smoke_ratio_distribution_{split}.png")
    
    # Show statistics
    print(f"\n=== Smoke Ratio Statistics ===")
    print(f"Min: {np.min(ratios):.4f}")
    print(f"Max: {np.max(ratios):.4f}")
    print(f"Mean: {np.mean(ratios):.4f}")
    print(f"Median: {np.median(ratios):.4f}")
    print(f"Std: {np.std(ratios):.4f}")
    print(f"\n=== Percentiles ===")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(ratios, p):.4f}")
    
    # Visualize examples at different thresholds
    print("\n=== Visualizing Examples ===")
    
    # Select examples: lowest, and at various percentiles
    indices = [
        0,  # Lowest
        int(len(video_ratios) * 0.01),  # 1st percentile
        int(len(video_ratios) * 0.05),  # 5th percentile
        int(len(video_ratios) * 0.10),  # 10th percentile
        int(len(video_ratios) * 0.25),  # 25th percentile
        int(len(video_ratios) * 0.50),  # 50th percentile
        int(len(video_ratios) * 0.75),  # 75th percentile
        int(len(video_ratios) * 0.90),  # 90th percentile
        len(video_ratios) - 1,  # Highest
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    labels = ['Lowest (0%)', '1st %ile', '5th %ile', '10th %ile', 
              '25th %ile', 'Median', '75th %ile', '90th %ile', 'Highest']
    
    for idx, (ax, label) in enumerate(zip(axes, labels)):
        video_id, ratio, img_path = video_ratios[indices[idx]]
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'{label}\nRatio: {ratio:.4f} ({ratio*100:.2f}%)', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'smoke_threshold_examples_{split}.png', dpi=150)
    print(f"Saved examples to smoke_threshold_examples_{split}.png")
    
    # Count how many would be filtered at different thresholds
    print(f"\n=== Filtering Impact ===")
    for min_thresh in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
        filtered_count = sum(1 for r in ratios if r < min_thresh)
        print(f"Min ratio {min_thresh:.3f} ({min_thresh*100:.1f}%): would filter {filtered_count}/{len(ratios)} "
              f"({filtered_count/len(ratios)*100:.1f}%) videos")


def main():
    parser = argparse.ArgumentParser(description="Visualize smoke ratio distribution")
    parser.add_argument("--data-root", type=str, default="final_dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=['train', 'val', 'test'],
                        help="Which split to analyze")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of videos to sample for analysis")
    
    args = parser.parse_args()
    
    visualize_smoke_thresholds(
        data_root=args.data_root,
        split=args.split,
        num_samples=args.num_samples
    )


if __name__ == "__main__":
    main()
