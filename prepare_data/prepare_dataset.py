import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

try:
    from pymatting import estimate_alpha_cf
except ImportError:
    estimate_alpha_cf = None

def generate_optimized_smoke_dataset(input_dir, output_dir, intensity_gain=1.2, show_monitor=True):
    video_paths = sorted(glob(os.path.join(input_dir, "*.mp4")))
    if not video_paths:
        raise FileNotFoundError(f"No MP4 videos found in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extraction : {len(video_paths)} vidéos trouvées.")

    if estimate_alpha_cf is None:
        print("Warning: 'pymatting' is not installed. Falling back to trimap-only alpha extraction.")

    monitor_opened = False
    for v_path in tqdm(video_paths):
        video_name = os.path.basename(v_path).split('.')[0]
        video_output_path = os.path.join(output_dir, video_name)

        if os.path.exists(video_output_path):
            if len(os.listdir(video_output_path)) >= 30:
                continue
        
        os.makedirs(video_output_path, exist_ok=True)

        cap = cv2.VideoCapture(v_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            # frame est chargé en BGR par cap.read()
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()

        if len(frames) < 5: continue

        video_stack = np.stack(frames)
        background = np.median(video_stack, axis=0)
        std_dev = np.std(video_stack, axis=0)
        turbulence = np.max(std_dev, axis=2)
        turbulence = cv2.normalize(turbulence, None, 0, 1, cv2.NORM_MINMAX)

        for i, frame in enumerate(frames):
            diff = np.max(np.abs(frame - background), axis=2)
            smoke_score = (diff * 0.7) + (turbulence * 0.3)
            max_s = np.max(smoke_score)

            if max_s < 0.06:
                continue 

            h, w = smoke_score.shape
            trimap = np.full((h, w), 0.5, dtype=np.float32)
            trimap[smoke_score < (max_s * 0.20)] = 0.0  
            trimap[smoke_score > (max_s * 0.65)] = 1.0  

            if np.sum(trimap == 1.0) < 10 or np.sum(trimap == 0.0) < 10:
                continue

            try:
                if estimate_alpha_cf is not None:
                    small_frame = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
                    small_trimap = cv2.resize(trimap, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
                    alpha_small = estimate_alpha_cf(
                        small_frame.astype(np.float64),
                        small_trimap.astype(np.float64),
                    )
                    alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    alpha = trimap.copy()
                    alpha[alpha == 0.5] = np.clip(smoke_score[alpha == 0.5] / (max_s + 1e-8), 0.0, 1.0)

                alpha = np.where(alpha < 0.1, 0, alpha)

                alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                
                # frame est en BGR -> extracted_smoke sera en BGR
                extracted_smoke = np.clip(frame * alpha_3d * intensity_gain, 0, 1)
                
                save_img = (extracted_smoke * 255).astype(np.uint8)
                
                # --- FIX : cv2.imwrite attend du BGR. save_img EST déjà en BGR. ---
                cv2.imwrite(os.path.join(video_output_path, f"frame_{i:04d}.png"), save_img)

                if show_monitor and i % 5 == 0:
                    # --- FIX : cv2.imshow attend aussi du BGR ---
                    cv2.imshow('MONITOR', save_img)
                    monitor_opened = True
                    if cv2.waitKey(1) == 27: show_monitor = False

            except Exception:
                continue
    
    if monitor_opened:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Extract smoke frames from MP4 videos.")
    parser.add_argument("--input-dir", default="smoke_videos")
    parser.add_argument("--output-dir", default="frames")
    parser.add_argument("--intensity-gain", type=float, default=1.3)
    parser.add_argument("--show-monitor", action="store_true", help="Open an OpenCV preview window")
    parser.add_argument("--require-matting", action="store_true", help="Fail instead of using the trimap fallback")
    args = parser.parse_args()
    if args.require_matting and estimate_alpha_cf is None:
        parser.error("pymatting is required; install it before preparing the dataset")
    generate_optimized_smoke_dataset(
        args.input_dir, args.output_dir,
        intensity_gain=args.intensity_gain, show_monitor=args.show_monitor,
    )


if __name__ == "__main__":
    main()
