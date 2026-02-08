"""
Test Flow Matching model with EXACT Wandb training visualization reproduction.
This script reproduces EXACTLY what happens during training for Wandb logging:
- Uses same generation code (trainer.py line 160-166)
- Uses same to_video() and make_observations_grid() functions
- Outputs both video and PNG grid matching Wandb style
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pymatting import estimate_alpha_cf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lutils.configuration import Configuration
from lutils.logging import to_video, make_observations_grid
from model.model import Model


def load_video_frames(video_path, max_frames=None, target_size=None):
    """Load frames from video file."""
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if target size specified
        if target_size is not None:
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    return frames


def extract_smoke_from_frames(frames, intensity_gain=1.3):
    """
    Extract foreground smoke from video frames using matting (from prepare_dataset.py).
    Returns: (extracted_frames, background, alpha_masks)
    """
    print("   Extracting smoke foreground...")
    
    # Convert to float [0, 1]
    frames_float = [f.astype(np.float32) / 255.0 for f in frames]
    video_stack = np.stack(frames_float)
    
    # Calculate background (median) - this is the clean background without smoke
    background = np.median(video_stack, axis=0)
    
    # Calculate turbulence (std)
    std_dev = np.std(video_stack, axis=0)
    turbulence = np.max(std_dev, axis=2)
    turbulence = cv2.normalize(turbulence, None, 0, 1, cv2.NORM_MINMAX)
    
    extracted_frames = []
    alpha_masks = []
    
    for i, frame in enumerate(frames_float):
        # Calculate smoke score
        diff = np.max(np.abs(frame - background), axis=2)
        smoke_score = (diff * 0.7) + (turbulence * 0.3)
        max_s = np.max(smoke_score)
        
        h, w = smoke_score.shape
        trimap = np.full((h, w), 0.5, dtype=np.float32)
        trimap[smoke_score < (max_s * 0.20)] = 0.0
        trimap[smoke_score > (max_s * 0.65)] = 1.0
        
        try:
            # Matting on smaller resolution for speed
            small_frame = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
            small_trimap = cv2.resize(trimap, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            
            alpha_small = estimate_alpha_cf(small_frame.astype(np.float64),
                                           small_trimap.astype(np.float64))
            
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = np.where(alpha < 0.1, 0, alpha)
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            # Extract smoke (on black background)
            extracted_smoke = np.clip(frame * alpha_3d * intensity_gain, 0, 1)
            extracted_frames.append((extracted_smoke * 255).astype(np.uint8))
            alpha_masks.append(alpha_3d)
            
        except Exception as e:
            # Fallback: use original frame if matting fails
            print(f"   Warning: Matting failed for frame {i}, using simple diff")
            simple_fg = np.clip((frame - background + 0.5) * intensity_gain, 0, 1)
            extracted_frames.append((simple_fg * 255).astype(np.uint8))
            alpha_masks.append(np.ones_like(frame))  # Full alpha as fallback
    
    print(f"   Extracted {len(extracted_frames)} smoke frames")
    # Return background as uint8 [0, 255]
    return extracted_frames, (background * 255).astype(np.uint8), alpha_masks


def frames_to_tensor(frames):
    """Convert list of numpy frames [0,255] to tensor [-1,1]."""
    # Stack to numpy array: [T, H, W, C]
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    
    # Convert to tensor and rearrange to [T, C, H, W]
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
    
    # Normalize to [-1, 1]
    frames_tensor = frames_tensor * 2.0 - 1.0
    
    return frames_tensor


def save_video_from_tensor(video_tensor, output_path, fps=7):
    """
    Save video tensor to file using EXACT same conversion as Wandb.
    Uses to_video() function from lutils.logging.
    
    Args:
        video_tensor: [T, C, H, W] in range [-1, 1]
        output_path: Path to save video
        fps: Frames per second
    """
    # Use EXACT same function as training
    video_np = to_video(video_tensor.unsqueeze(0))  # Add batch dim
    # to_video returns [B, T, C, H, W] in uint8
    video_np = video_np[0]  # Remove batch dim: [T, C, H, W]
    
    # Permute to [T, H, W, C] for OpenCV
    T, C, H, W = video_np.shape
    video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, C]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    
    # Write frames
    for t in range(T):
        frame_bgr = cv2.cvtColor(video_np[t], cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def compose_smoke_with_background(smoke_tensor, background_np, alpha=0.8):
    """
    Compose smoke (foreground in [-1,1]) with background.
    
    Args:
        smoke_tensor: [T, C, H, W] in range [-1, 1] (generated smoke)
        background_np: [H, W, C] in range [0, 255] (clean background)
        alpha: blending factor for smoke
    Returns:
        composed_frames: [T, C, H, W] in range [-1, 1]
    """
    T, C, H, W = smoke_tensor.shape
    
    # Convert smoke from [-1, 1] to [0, 1]
    smoke_01 = (smoke_tensor + 1.0) / 2.0
    
    # Convert background to tensor [0, 1]
    background_01 = torch.from_numpy(background_np.astype(np.float32) / 255.0)
    background_01 = background_01.permute(2, 0, 1)  # [C, H, W]
    
    # Resize background if needed
    if background_01.shape[1] != H or background_01.shape[2] != W:
        background_01 = torch.nn.functional.interpolate(
            background_01.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False)[0]
    
    # Expand background for all frames
    background_batch = background_01.unsqueeze(0).expand(T, -1, -1, -1)  # [T, C, H, W]
    
    # Simple alpha compositing: result = smoke * alpha + background * (1 - alpha)
    # But smoke on black background, so we use additive blending
    composed = torch.clamp(smoke_01 + background_batch * 0.7, 0, 1)
    
    # Convert back to [-1, 1]
    composed = composed * 2.0 - 1.0
    
    return composed


def save_grid_from_tensors(gt_observations, pred_observations, output_path):
    """
    Create a clean comparison grid with all frames: GT on top, Pred on bottom.
    Shows "GT" label on top and "MSE PSNR" scores below each frame pair.
    
    Args:
        gt_observations: [1, T, C, H, W] ground truth in [-1, 1]
        pred_observations: [1, T, C, H, W] predictions in [-1, 1]
        output_path: Path to save PNG
    """
    # Remove batch dimension
    gt = gt_observations[0]  # [T, C, H, W]
    pred = pred_observations[0]  # [T, C, H, W]
    
    T, C, H, W = gt.shape
    
    # Convert to [0, 255] uint8
    gt_uint8 = ((torch.clamp(gt, -1, 1) + 1) / 2 * 255).byte()
    pred_uint8 = ((torch.clamp(pred, -1, 1) + 1) / 2 * 255).byte()
    
    # Calculate MSE and PSNR for each frame
    mse_list = []
    psnr_list = []
    for t in range(T):
        mse = torch.nn.functional.mse_loss(pred[t], gt[t]).item()
        psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
        mse_list.append(mse)
        psnr_list.append(psnr)
    
    # Create grid with text annotations
    text_height = 20  # Height for text at top
    score_height = 25  # Height for scores at bottom
    
    columns = []
    for t in range(T):
        gt_frame = gt_uint8[t].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        pred_frame = pred_uint8[t].permute(1, 2, 0).cpu().numpy()
        
        # Create column: text + GT + Pred + scores
        column_height = text_height + H * 2 + score_height
        column = np.ones((column_height, W, 3), dtype=np.uint8) * 255
        
        # Add "GT" text at top (centered)
        if t == T // 2:  # Only on middle frame
            cv2.putText(column, "GT", (W // 2 - 10, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Add GT frame
        column[text_height:text_height + H] = gt_frame
        
        # Add Pred frame
        column[text_height + H:text_height + H * 2] = pred_frame
        
        # Add scores at bottom
        mse_val = mse_list[t]
        psnr_val = psnr_list[t]
        score_text = f"MSE:{mse_val:.3f} PSNR:{psnr_val:.1f}"
        
        # Calculate text size and position to center it
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        text_x = max(0, (W - text_size[0]) // 2)
        text_y = text_height + H * 2 + 15
        
        cv2.putText(column, score_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        columns.append(column)
    
    # Stack all columns horizontally
    grid = np.hstack(columns)
    
    # Save as PNG
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Test Flow Matching with EXACT Wandb reproduction")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--output-video", type=str, default="wandb_exact_video.mp4", 
                       help="Output video path")
    parser.add_argument("--output-grid", type=str, default="wandb_exact_grid.png",
                       help="Output grid PNG path")
    parser.add_argument("--num-videos", type=int, default=1,
                       help="Number of videos to generate (default 1 for full video generation)")
    parser.add_argument("--fps", type=int, default=7,
                       help="FPS for output video (default 7 like Wandb)")
    parser.add_argument("--num-condition", type=int, default=10,
                       help="Number of condition frames to use (default 10)")
    parser.add_argument("--max-generate", type=int, default=None,
                       help="Max frames to generate (None = all remaining frames)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎬 EXACT WANDB REPRODUCTION TEST")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Video: {args.video}")
    print(f"Output video: {args.output_video}")
    print(f"Output grid: {args.output_grid}")
    print()
    
    # Load config
    print("📄 Loading config...")
    config = Configuration(args.config)
    num_condition_frames = args.num_condition
    target_size = config["data"]["input_size"]
    
    print(f"   Condition frames: {num_condition_frames}")
    print(f"   Target size: {target_size}x{target_size}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    # Load model
    print("\n🤖 Loading model...")
    model = Model(config["model"])
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"   Loaded from step: {checkpoint.get('global_step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("   Model loaded successfully!")
    
    # Load video
    print(f"\n🎥 Loading ALL video frames...")
    all_frames = load_video_frames(args.video, target_size=target_size)
    
    print(f"   Loaded {len(all_frames)} frames total")
    print(f"   Resized to: {target_size}x{target_size}")
    
    # Extract smoke (foreground) from frames
    print(f"\n🔥 Extracting smoke foreground...")
    all_frames, background, alpha_masks = extract_smoke_from_frames(all_frames, intensity_gain=1.3)
    print(f"   ✅ Smoke extraction complete")
    
    if len(all_frames) < num_condition_frames + 1:
        print(f"   ❌ ERROR: Not enough frames! Need at least {num_condition_frames + 1}, got {len(all_frames)}")
        return
    
    # Determine how many frames to generate
    max_frames_available = len(all_frames) - num_condition_frames
    if args.max_generate is not None:
        frames_to_generate = min(args.max_generate, max_frames_available)
    else:
        frames_to_generate = max_frames_available
    
    print(f"   Will use {num_condition_frames} condition frames")
    print(f"   Will generate {frames_to_generate} frames")
    
    # Prepare data
    print(f"\n🔄 Preparing sequence...")
    all_frames_tensor = frames_to_tensor(all_frames[:num_condition_frames + frames_to_generate])  # [T, C, H, W]
    
    condition = all_frames_tensor[:num_condition_frames].unsqueeze(0).to(device)  # [1, T_cond, C, H, W]
    ground_truth = all_frames_tensor.unsqueeze(0)  # [1, T_total, C, H, W]
    
    print(f"   Condition: {condition.shape}")
    print(f"   Ground truth (full): {ground_truth.shape}")
    
    # Generate frames using flexible conditioning
    print(f"\n🎨 Generating {frames_to_generate} frames...")
    print(f"   Conditioning frames: {num_condition_frames} frames")
    print(f"   Note: Model will randomly sample from these {num_condition_frames} frames during generation")
    print(f"   Reference frame: last frame of conditioning (frame {num_condition_frames})")
    
    with torch.no_grad():
        # You can use ANY number of condition frames!
        # The model will:
        # 1. Use the LAST frame as reference
        # 2. RANDOMLY sample from all condition frames for conditioning
        # This is more flexible than training (which used condition_frames: 1)
        
        generation_condition = condition  # [1, num_condition, C, H, W]
        
        print(f"   Condition for generation: {generation_condition.shape}")
        print(f"   Generating: {frames_to_generate} frames in one shot")
        print(f"   ODE steps: 100")
        
        # Generate ALL frames at once
        generated_full = model.generate_frames(
            observations=generation_condition,
            num_frames=frames_to_generate,
            steps=100,
            verbose=True
        )  # Returns [1, T_cond + T_gen, C, H, W] - includes condition frames!
    
    print(f"   ✅ Generated (full): {generated_full.shape}")
    
    # Extract ONLY the generated frames (not the condition frames)
    generated = generated_full[:, num_condition_frames:].cpu()  # [1, T_gen, C, H, W]
    
    # Extract ground truth generated frames
    gt_generated = ground_truth[:, num_condition_frames:].cpu()  # [1, T_gen, C, H, W]
    
    print(f"\n   Ground truth (generated frames only): {gt_generated.shape}")
    print(f"   Generated: {generated.shape}")
    
    # Save video using EXACT same to_video() function
    print(f"\n🎬 Creating video (EXACT Wandb style)...")
    print(f"   Using to_video() from lutils.logging")
    print(f"   FPS: {args.fps}")
    
    # Video of generated frames
    video_tensor = generated[0]  # [T_gen, C, H, W]
    save_video_from_tensor(video_tensor, args.output_video, fps=args.fps)
    print(f"   ✅ Video saved: {args.output_video}")
    
    # Save grid using EXACT same make_observations_grid() function
    print(f"\n🖼️  Creating grid (EXACT Wandb style)...")
    print(f"   Using make_observations_grid() from lutils.logging")
    print(f"   Comparing {frames_to_generate} generated frames vs ground truth")
    
    # Grid: Row 1 = Ground truth, Row 2 = Generated
    save_grid_from_tensors(gt_generated, generated, args.output_grid)
    print(f"   ✅ Grid saved: {args.output_grid}")
    
    # Calculate metrics
    print(f"\n📊 Metrics:")
    mse = torch.nn.functional.mse_loss(generated, gt_generated).item()
    print(f"   Overall MSE: {mse:.6f}")
    
    # MSE per 10-frame chunks
    chunk_size = 10
    num_chunks = (frames_to_generate + chunk_size - 1) // chunk_size
    print(f"\n   MSE per {chunk_size}-frame chunks:")
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, frames_to_generate)
        
        gen_chunk = generated[:, start:end]
        gt_chunk = gt_generated[:, start:end]
        
        chunk_mse = torch.nn.functional.mse_loss(gen_chunk, gt_chunk).item()
        print(f"      Frames {start+1}-{end}: MSE = {chunk_mse:.6f}")
    
    print("\n" + "=" * 80)
    print("📝 SUMMARY:")
    print(f"   Conditioning: {num_condition_frames} frames (flexible, not limited to config)")
    print(f"   Generated: {frames_to_generate} frames in one shot")
    print(f"   ODE steps: 100")
    print(f"   Model behavior:")
    print(f"      - Uses last condition frame as reference")
    print(f"      - Randomly samples from all {num_condition_frames} frames for conditioning")
    print(f"   Overall MSE: {mse:.6f}")
    print(f"   Conversion: to_video() [-1,1] → [0,255]")
    print(f"   Grid: make_observations_grid() stacks GT + generated")
    print(f"   Output video: {args.output_video}")
    print(f"   Output grid: {args.output_grid}")
    print("=" * 80)


if __name__ == "__main__":
    main()
