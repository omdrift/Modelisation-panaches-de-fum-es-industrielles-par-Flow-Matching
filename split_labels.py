import os
import cv2
import numpy as np
import random
from pymatting import estimate_alpha_cf
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def get_dark_channel(img, size=15):
    """Estimates the atmospheric 'thickness' to see through fog[cite: 256, 386]."""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def process_video_dcp(args):
    v_path, output_split_dir, v_name = args
    os.makedirs(output_split_dir, exist_ok=True)

    cap = cv2.VideoCapture(v_path)
    frames = []
    # 1. Establish the 'Clean Plate' for THIS specific video 
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample frames across the video to build a median background
    sample_idx = np.linspace(0, total_f - 1, 40, dtype=int)
    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret: frames.append(f.astype(np.float64) / 255.0)
    
    if len(frames) < 10: return []
    
    # Calculate video-specific background and its dark channel [cite: 109, 212]
    bg_img = np.median(np.stack(frames), axis=0)
    bg_dark = get_dark_channel(bg_img)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    local_manifest = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        # Skip frames to reduce redundancy for VQGAN (Industrial smoke is slow) [cite: 304]
        if frame_idx % 4 != 0:
            frame_idx += 1
            continue

        img = frame.astype(np.float64) / 255.0
        
        # 2. Compare the current Dark Channel to the Background Dark Channel
        # This isolates the plume even if the whole scene is blue/gray/foggy [cite: 98, 387]
        frame_dark = get_dark_channel(img)
        diff = np.abs(frame_dark - bg_dark)
        diff = cv2.GaussianBlur(diff, (7, 7), 0)

        # 3. Generate the Trimap for Matting [cite: 104, 250]
        h, w = diff.shape
        trimap = np.full((h, w), 0.5)
        # Higher thresholds for foggy videos to ensure rigor
        trimap[diff < 0.12] = 0.0 # Confirmed Background
        trimap[diff > 0.35] = 1.0 # Confirmed Smoke Core

        # Filter out frames where smoke isn't detected [cite: 243, 261]
        if np.sum(trimap == 1.0) < 50:
            frame_idx += 1
            continue

        try:
            # 4. Closed-Form Matting for high-quality alpha [cite: 109, 256]
            # We scale down for speed; CF Matting is mathematically intensive [cite: 109]
            scale = 2
            small_img = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
            small_tri = cv2.resize(trimap, (w//scale, h//scale), interpolation=cv2.INTER_NEAREST)
            
            alpha_small = estimate_alpha_cf(small_img, small_tri)
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = np.clip(alpha, 0, 1)

            # 5. Export as RGBA (Smoke on Black Background) [cite: 23, 63]
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            # Black-out background pixels by multiplying with Alpha [cite: 23, 98]
            rgba[:, :, :3] = (frame * alpha[:, :, np.newaxis]).astype(np.uint8)
            rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

            save_name = f"{v_name}_f{frame_idx:04d}.png"
            cv2.imwrite(os.path.join(output_split_dir, save_name), rgba)
            local_manifest.append(save_name.replace('.png', ''))
        except:
            pass
        
        frame_idx += 1
    cap.release()
    return local_manifest

def run_rigorous_pipeline(input_root, final_root, splits=(0.8, 0.1, 0.1)):
    # Group by view_id folders to prevent background leakage 
    view_folders = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    random.seed(42)
    random.shuffle(view_folders)

    n = len(view_folders)
    split_map = {
        'train': view_folders[:int(n*splits[0])],
        'val': view_folders[int(n*splits[0]):int(n*(splits[0]+splits[1]))],
        'test': view_folders[int(n*(splits[0]+splits[1])):]
    }

    for split_name, views in split_map.items():
        print(f"--- Processing Split: {split_name} ---")
        split_dir = os.path.join(final_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        tasks = []
        for v_id in views:
            v_folder = os.path.join(input_root, v_id)
            for v_file in os.listdir(v_folder):
                if v_file.endswith(".mp4"):
                    tasks.append((os.path.join(v_folder, v_file), split_dir, v_file.split('.')[0]))

        # Use all CPU cores to process 5,042 videos quickly [cite: 236]
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_video_dcp, tasks), total=len(tasks)))

        # Save manifest for VQGAN loader [cite: 259]
        all_frames = [item for sublist in results for item in sublist]
        with open(os.path.join(final_root, f"{split_name}.txt"), "w") as f:
            f.write("\n".join(all_frames))

if __name__ == "__main__":
    # Ensure smoke_videos contains your view_X-Y subfolders
    run_rigorous_pipeline("smoke_videos", "final_dataset")