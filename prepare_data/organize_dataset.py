import os
import shutil
from tqdm import tqdm

def organize_by_view_id(smoke_videos_dir):
    if not os.path.exists(smoke_videos_dir):
        print(f"Error: {smoke_videos_dir} not found.")
        return

    video_files = [f for f in os.listdir(smoke_videos_dir) if f.endswith('.mp4')]
    print(f"Organizing {len(video_files)} videos...")

    for filename in tqdm(video_files):
        try:
            # 1. Split at the first underscore: '174_0-0-2018...' -> ['174', '0-0-2018...']
            after_first_underscore = filename.split('_', 1)[1]
            
            # 2. Split that result by hyphens to get the first two parts: '0' and '0'
            parts = after_first_underscore.split('-')
            view_id = f"{parts[0]}-{parts[1]}" # This creates '0-0', '2-1', etc.
            
            # 3. Define the folder based on the view_id
            target_folder = os.path.join(smoke_videos_dir, f"view_{view_id}")
            
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            
            # 4. Move the video
            shutil.move(os.path.join(smoke_videos_dir, filename), 
                        os.path.join(target_folder, filename))
                        
        except (IndexError, ValueError):
            print(f"Skipping file with naming anomaly: {filename}")
            continue

    print("\nFolders organized by View ID.")

if __name__ == "__main__":

    target_path = '/home/aoubaidi/Documents/Modelisation-panaches-de-fum-es-industrielles-par-Flow-Matching/smoke_videos'

    organize_by_view_id(target_path)
    # Ensure this path matches your environment
