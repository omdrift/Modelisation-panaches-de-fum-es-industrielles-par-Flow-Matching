import os
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Load the RVM model
# We use ResNet50 for higher precision on wispy smoke textures
model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50")
model.load_state_dict(torch.load("rvm_resnet50.pth"))
model = model.eval().cuda() # Ensure you have a GPU

def process_smoke_dataset(input_dir, output_root):
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # Create output directories
    os.makedirs(output_root, exist_ok=True)

    for video_name in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(input_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        
        # Prepare output subfolder for this video
        video_id = os.path.splitext(video_name)[0]
        save_path = os.path.join(output_root, video_id)
        os.makedirs(save_path, exist_ok=True)

        # RVM needs to keep track of state across frames
        rec_state = [None] * 4 
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB and normalize to [0, 1]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

            # Inference
            with torch.no_grad():
                # fgr = foreground, pha = alpha matte
                fgr, pha, *rec_state = model(src, *rec_state, downsample_ratio=0.25)

            # Convert back to CPU/Numpy
            fgr = fgr.squeeze().permute(1, 2, 0).cpu().numpy()
            pha = pha.squeeze().cpu().numpy()

            # Create RGBA Image
            # Using the predicted foreground (fgr) is better than the original frame
            # because RVM cleans up the color at the edges of the alpha
            rgba = np.zeros((fgr.shape[0], fgr.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = (fgr * 255).astype(np.uint8)
            rgba[:, :, 3] = (pha * 255).astype(np.uint8)

            # Save frame
            img = Image.fromarray(rgba, 'RGBA')
            img.save(os.path.join(save_path, f"frame_{frame_idx:04d}.png"))
            frame_idx += 1

        cap.release()

if __name__ == "__main__":
    process_smoke_dataset('smoke_videos', 'final_dataset/raw_rgba')