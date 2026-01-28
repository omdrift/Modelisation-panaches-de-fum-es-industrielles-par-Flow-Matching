import os
import json
import re
from collections import defaultdict

def analyze_missing_frames(dataset_root, output_json="missing_frames_report.json"):
    report = {"train": {}, "test": {}}
    total_missing = 0

    for split in ["train", "test"]:
        split_path = os.path.join(dataset_root, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split}: Folder not found.")
            continue

        # Group files by Video ID
        # Pattern matches: [VideoID]_frame_[XXXX].png
        video_frames = defaultdict(list)
        files = [f for f in os.listdir(split_path) if f.endswith('.png')]
        
        for f in files:
            # Extract video_id and frame_index
            match = re.match(r"(.+)_frame_(\d+)\.png", f)
            if match:
                video_id = match.group(1)
                frame_idx = int(match.group(2))
                video_frames[video_id].append(frame_idx)

        print(f"\nAnalyzing {split} split...")
        for video_id, indices in video_frames.items():
            indices.sort()
            start, end = indices[0], indices[-1]
            
            # Check for gaps between 0 and the highest frame found
            # (Assuming every video MUST start at 0000)
            full_range = set(range(0, end + 1))
            actual_range = set(indices)
            missing = sorted(list(full_range - actual_range))
            
            if missing:
                report[split][video_id] = {
                    "count": len(missing),
                    "missing_indices": [f"{i:04d}" for i in missing],
                    "total_existing": len(indices),
                    "max_index_found": end
                }
                total_missing += len(missing)

    # Save to JSON
    with open(output_json, "w") as j:
        json.dump(report, j, indent=4)

    # Print Summary
    print("-" * 30)
    print(f"Analysis Complete!")
    print(f"Total missing frames detected: {total_missing}")
    print(f"Report saved to: {output_json}")
    
    if total_missing > 0:
        print("\nNote: Your DataLoader will fail on these videos unless you re-number them.")

# --- CONFIGURATION ---
DATASET_PATH = "/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/final_dataset/"

analyze_missing_frames(DATASET_PATH)