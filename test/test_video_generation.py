"""
Test Flow Matching: Generate future frames from 10 condition frames.
Uses pymatting to extract smoke/background, generates frames, and composes them back.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from pymatting import estimate_alpha_cf
from skimage.metrics import structural_similarity as ssim

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lutils.configuration import Configuration
from lutils.logging import to_video
from model.model import Model


def load_video_frames(video_path, target_size=64):
    """Load all frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
    
    cap.release()
    return frames


def extract_smoke_and_background(frames, intensity_gain=1.3):
    """
    Extract smoke (foreground) and clean background from video frames.
    Returns: (smoke_frames, background)
    """
    print("   Extracting smoke and background...")
    
    # Convert to float [0, 1]
    frames_float = [f.astype(np.float32) / 255.0 for f in frames]
    video_stack = np.stack(frames_float)
    
    # Calculate background (median) - clean background without smoke
    background = np.median(video_stack, axis=0)
    
    # Calculate turbulence
    std_dev = np.std(video_stack, axis=0)
    turbulence = np.max(std_dev, axis=2)
    turbulence = cv2.normalize(turbulence, None, 0, 1, cv2.NORM_MINMAX)
    
    smoke_frames = []
    
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
            # Matting
            small_frame = cv2.resize(frame, (w//2, h//2), interpolation=cv2.INTER_AREA)
            small_trimap = cv2.resize(trimap, (w//2, h//2), interpolation=cv2.INTER_NEAREST)
            
            alpha_small = estimate_alpha_cf(small_frame.astype(np.float64),
                                           small_trimap.astype(np.float64))
            
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
            alpha = np.where(alpha < 0.1, 0, alpha)
            alpha_3d = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            
            # Extract smoke on black background
            extracted_smoke = np.clip(frame * alpha_3d * intensity_gain, 0, 1)
            smoke_frames.append((extracted_smoke * 255).astype(np.uint8))
            
        except Exception as e:
            print(f"   Warning: Matting failed for frame {i}")
            simple_fg = np.clip((frame - background + 0.5) * intensity_gain, 0, 1)
            smoke_frames.append((simple_fg * 255).astype(np.uint8))
    
    print(f"   Extracted {len(smoke_frames)} smoke frames")
    return smoke_frames, (background * 255).astype(np.uint8)


def frames_to_tensor(frames):
    """Convert frames [0,255] to tensor [-1,1]."""
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
    return frames_tensor * 2.0 - 1.0


def tensor_to_frames(tensor):
    """Convert tensor [-1,1] to frames [0,255]."""
    # tensor: [T, C, H, W]
    frames = ((torch.clamp(tensor, -1, 1) + 1) / 2 * 255).byte()
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    return frames


def compose_smoke_with_background(smoke_frames, background):
    """
    Compose smoke frames with background using additive blending.
    smoke_frames: [T, H, W, C] uint8
    background: [H, W, C] uint8
    """
    composed = []
    for smoke in smoke_frames:
        # Simple additive blending
        smoke_float = smoke.astype(np.float32) / 255.0
        bg_float = background.astype(np.float32) / 255.0
        
        # Add smoke to background (with some attenuation for realism)
        result = np.clip(smoke_float + bg_float * 0.7, 0, 1)
        composed.append((result * 255).astype(np.uint8))
    
    return composed


def calculate_metrics(gt_frame, pred_frame):
    """
    Calculate MSE, PSNR, SSIM between two frames.
    Uses standard definitions for rigorous evaluation.
    
    Args:
        gt_frame: Ground truth frame [H, W, C] in [0, 255] uint8
        pred_frame: Predicted frame [H, W, C] in [0, 255] uint8
    
    Returns:
        mse: Mean Squared Error on [0, 255] scale
        psnr: Peak Signal-to-Noise Ratio in dB
        ssim_value: Structural Similarity Index in [0, 1]
    """
    # Ensure inputs are float32 for accurate computation
    gt = gt_frame.astype(np.float64)
    pred = pred_frame.astype(np.float64)
    
    # MSE - calculated on [0, 255] scale (standard)
    mse = np.mean((gt - pred) ** 2)
    
    # PSNR - standard formula: PSNR = 10 * log10(MAX^2 / MSE)
    # For 8-bit images, MAX = 255
    if mse > 0:
        max_pixel = 255.0
        psnr = 10.0 * np.log10((max_pixel ** 2) / mse)
    else:
        psnr = 100.0  # Perfect match
    
    # SSIM - calculated on grayscale for consistency
    # Convert to grayscale (using standard weights)
    gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2GRAY)
    pred_gray = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2GRAY)
    
    # Calculate SSIM with proper data_range for uint8 images
    ssim_value = ssim(gt_gray, pred_gray, data_range=255)
    
    return mse, psnr, ssim_value


def create_label_bar(text, width, bg_color=(50, 50, 50), text_color=(255, 255, 255)):
    """Create a clean label bar."""
    bar = np.full((30, width, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - text_width) // 2
    y = (30 + text_height) // 2
    
    cv2.putText(bar, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return bar


def create_metrics_bar(mse, psnr, ssim_val, width):
    """Create a clean metrics display bar with proper MSE formatting."""
    bar_height = 50
    bar = np.full((bar_height, width, 3), (30, 30, 30), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    
    # Determine color based on quality (green = good, yellow = medium, red = bad)
    # PSNR thresholds: >30dB=excellent, >20dB=good, >15dB=acceptable
    psnr_color = (0, 255, 0) if psnr >= 30 else (0, 200, 255) if psnr >= 20 else (0, 100, 255)
    ssim_color = (0, 255, 0) if ssim_val >= 0.8 else (0, 200, 255) if ssim_val >= 0.6 else (0, 100, 255)
    
    # MSE formatting: if very large, use scientific notation
    if mse >= 1000:
        mse_text = f"MSE: {mse:.2e}"
    else:
        mse_text = f"MSE: {mse:.2f}"
    
    # Create metric texts
    y_start = 17
    metrics = [
        (mse_text, (180, 180, 180)),
        (f"PSNR: {psnr:.2f}dB", psnr_color),
        (f"SSIM: {ssim_val:.3f}", ssim_color)
    ]
    
    # Center all metrics
    total_width = sum([cv2.getTextSize(m[0], font, font_scale, thickness)[0][0] for m in metrics]) + 40
    x_start = (width - total_width) // 2
    
    for text, color in metrics:
        cv2.putText(bar, text, (x_start, y_start), font, font_scale, color, thickness, cv2.LINE_AA)
        text_width = cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        x_start += text_width + 20
        
        y_start2 = y_start + 18
        if text.startswith("MSE"):
            cv2.putText(bar, "(lower=better)", (x_start - text_width - 20, y_start2), 
                       font, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
        elif text.startswith("PSNR"):
            cv2.putText(bar, "(higher=better)", (x_start - text_width - 20, y_start2), 
                       font, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
        elif text.startswith("SSIM"):
            cv2.putText(bar, "(1.0=perfect)", (x_start - text_width - 20, y_start2), 
                       font, 0.3, (120, 120, 120), 1, cv2.LINE_AA)
    
    return bar


def save_comparison_grid(gt_frames, pred_frames, output_path, num_show=8):
    """
    Create professional comparison grid with clean layout.
    GT: Original frames (top)
    PRED: Composed predictions (bottom)
    Metrics: Clean bar below each column
    """
    T = len(gt_frames)
    # Select frames uniformly
    indices = np.linspace(0, T - 1, min(num_show, T)).astype(int)
    
    # Calculate metrics for selected frames
    all_metrics = []
    for idx in indices:
        mse, psnr, ssim_val = calculate_metrics(gt_frames[idx], pred_frames[idx])
        all_metrics.append((mse, psnr, ssim_val))
    
    # Create columns
    columns = []
    spacing = 5  # Spacing between columns
    
    for i, idx in enumerate(indices):
        gt = gt_frames[idx]
        pred = pred_frames[idx]
        h, w = gt.shape[:2]
        
        mse, psnr, ssim_val = all_metrics[i]
        
        # Create column components
        gt_label = create_label_bar("GROUND TRUTH", w, (0, 100, 0), (255, 255, 255))
        pred_label = create_label_bar("PREDICTION", w, (50, 50, 150), (255, 255, 255))
        metrics_bar = create_metrics_bar(mse, psnr, ssim_val, w)
        frame_number = create_label_bar(f"Frame {idx + 1}", w, (70, 70, 70), (200, 200, 200))
        
        # Stack vertically: Frame number, GT label, GT image, Pred label, Pred image, Metrics
        column = np.vstack([
            frame_number,
            gt_label,
            gt,
            np.full((10, w, 3), (20, 20, 20), dtype=np.uint8),  # Spacer
            pred_label,
            pred,
            metrics_bar
        ])
        
        # Add spacing on the right (except last column)
        if i < len(indices) - 1:
            spacer = np.full((column.shape[0], spacing, 3), (0, 0, 0), dtype=np.uint8)
            column = np.hstack([column, spacer])
        
        columns.append(column)
    
    # Stack horizontally
    grid = np.hstack(columns)
    
    # Create title bar
    title_height = 60
    title_bar = np.full((title_height, grid.shape[1], 3), (40, 40, 40), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate average metrics
    avg_mse = np.mean([m[0] for m in all_metrics])
    avg_psnr = np.mean([m[1] for m in all_metrics])
    avg_ssim = np.mean([m[2] for m in all_metrics])
    
    # Main title
    title_text = "SMOKE GENERATION EVALUATION"
    cv2.putText(title_bar, title_text, (20, 25), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Subtitle with average metrics
    subtitle = f"Frames {indices[0]+1}-{indices[-1]+1}  |  Avg: MSE={avg_mse:.4f}  PSNR={avg_psnr:.2f}dB  SSIM={avg_ssim:.3f}"
    cv2.putText(title_bar, subtitle, (20, 50), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Quality indicator
    quality_text = "Quality: "
    if avg_psnr >= 30 and avg_ssim >= 0.85:
        quality_text += "EXCELLENT"
        quality_color = (0, 255, 0)
    elif avg_psnr >= 20 and avg_ssim >= 0.7:
        quality_text += "GOOD"
        quality_color = (0, 200, 255)
    elif avg_psnr >= 15:
        quality_text += "ACCEPTABLE"
        quality_color = (0, 150, 255)
    else:
        quality_text += "NEEDS IMPROVEMENT"
        quality_color = (0, 100, 255)
    
    text_width = cv2.getTextSize(quality_text, font, 0.5, 1)[0][0]
    cv2.putText(title_bar, quality_text, (grid.shape[1] - text_width - 20, 50), 
                font, 0.5, quality_color, 2, cv2.LINE_AA)
    
    # Combine title and grid
    final_grid = np.vstack([title_bar, grid])
    
    # Add bottom info bar
    bottom_bar = np.full((25, grid.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    info_text = "Note: PSNR >30dB=excellent, >20dB=good, >15dB=acceptable | SSIM >0.8=good | MSE on [0,255] scale"
    cv2.putText(bottom_bar, info_text, (20, 17), font, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    final_grid = np.vstack([final_grid, bottom_bar])
    
    cv2.imwrite(str(output_path), cv2.cvtColor(final_grid, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Grid saved: {output_path}")
    print(f"   📊 Average metrics: MSE={avg_mse:.4f}, PSNR={avg_psnr:.2f} dB, SSIM={avg_ssim:.3f}")
    print(f"   💡 Quality assessment: {quality_text.split(': ')[1]}")
    if avg_psnr < 20 or avg_ssim < 0.8:
        print(f"   ⚠️  Model performance is below optimal (target: PSNR>30dB for excellent, >20dB for good)")


def main():
    parser = argparse.ArgumentParser(description="Generate future smoke frames")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output-grid", type=str, default="prediction_grid.png")
    parser.add_argument("--num-condition", type=int, default=10)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎬 SMOKE FLOW MATCHING TEST - Full Video Generation")
    print("=" * 80)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Condition frames: {args.num_condition}")
    print()
    
    # Load config
    config = Configuration(args.config)
    target_size = config["data"]["input_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("🤖 Loading model...")
    model = Model(config["model"])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print(f"   Step: {checkpoint.get('global_step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    # Load video (36 frames)
    print(f"\n🎥 Loading video frames...")
    all_frames = load_video_frames(args.video, target_size=target_size)
    print(f"   Loaded {len(all_frames)} frames at {target_size}x{target_size}")
    
    if len(all_frames) != 36:
        print(f"   Warning: Expected 36 frames, got {len(all_frames)}")
    
    # Extract smoke and background
    print(f"\n🔥 Extracting smoke and background...")
    smoke_frames, background = extract_smoke_and_background(all_frames, intensity_gain=1.3)
    print(f"   Background extracted (clean image without smoke)")
    
    # Take first 10 frames as condition
    num_cond = args.num_condition
    num_gen = len(smoke_frames) - num_cond  # Generate remaining frames (26 if total=36)
    
    condition_frames = smoke_frames[:num_cond]
    gt_smoke_frames = smoke_frames[num_cond:]  # GT smoke for generation
    gt_original_frames = all_frames[num_cond:]  # Original frames for visualization
    
    print(f"\n📊 Setup:")
    print(f"   Condition: frames 1-{num_cond}")
    print(f"   Generate: frames {num_cond+1}-{len(smoke_frames)} ({num_gen} frames)")
    
    # Convert to tensors
    condition_tensor = frames_to_tensor(condition_frames).unsqueeze(0).to(device)  # [1, T_cond, C, H, W]
    
    # Generate future frames
    print(f"\n🎨 Generating {num_gen} frames...")
    with torch.no_grad():
        generated_full = model.generate_frames(
            observations=condition_tensor,
            num_frames=num_gen,
            steps=100,
            verbose=True
        )  # Returns [1, T_cond + T_gen, C, H, W]
    
    # Extract only generated frames (remove condition)
    generated_tensor = generated_full[0, num_cond:].cpu()  # [T_gen, C, H, W]
    print(f"   Generated: {generated_tensor.shape}")
    
    # Convert to frames
    generated_smoke_frames = tensor_to_frames(generated_tensor)
    
    # Compose predictions with background
    print(f"\n🖼️  Composing predictions with background...")
    pred_composed = compose_smoke_with_background(generated_smoke_frames, background)
    
    # Save comparison grid (GT = original frames, PRED = composed predictions)
    print(f"\n💾 Saving comparison grid with metrics...")
    save_comparison_grid(gt_original_frames, pred_composed, args.output_grid, num_show=min(num_gen, 10))
    
    # Calculate overall metrics (smoke-only comparison for training metric)
    gt_tensor = frames_to_tensor(gt_smoke_frames)
    mse_model = torch.nn.functional.mse_loss(generated_tensor, gt_tensor).item()
    
    # ===== BASELINE COMPARISONS =====
    print(f"\n📊 Baseline Comparisons:")
    print(f"   Comparing model predictions against simple baselines...")
    
    # Baseline 1: Repeat last condition frame
    last_condition_frame = condition_frames[-1]
    baseline_repeat = [last_condition_frame] * num_gen
    baseline_repeat_composed = compose_smoke_with_background(baseline_repeat, background)
    
    # Baseline 2: Mean of all condition frames
    mean_condition = np.mean(condition_frames, axis=0).astype(np.uint8)
    baseline_mean = [mean_condition] * num_gen
    baseline_mean_composed = compose_smoke_with_background(baseline_mean, background)
    
    # Calculate metrics for baselines on composed images (visual comparison)
    mse_repeat_list = []
    mse_mean_list = []
    mse_model_visual_list = []
    
    for i in range(num_gen):
        # Model
        mse_m, _, _ = calculate_metrics(gt_original_frames[i], pred_composed[i])
        mse_model_visual_list.append(mse_m)
        
        # Baseline: Repeat last frame
        mse_r, _, _ = calculate_metrics(gt_original_frames[i], baseline_repeat_composed[i])
        mse_repeat_list.append(mse_r)
        
        # Baseline: Mean frame
        mse_mean_val, _, _ = calculate_metrics(gt_original_frames[i], baseline_mean_composed[i])
        mse_mean_list.append(mse_mean_val)
    
    avg_mse_model = np.mean(mse_model_visual_list)
    avg_mse_repeat = np.mean(mse_repeat_list)
    avg_mse_mean = np.mean(mse_mean_list)
    
    # Calculate improvement percentages
    improvement_vs_repeat = ((avg_mse_repeat - avg_mse_model) / avg_mse_repeat) * 100
    improvement_vs_mean = ((avg_mse_mean - avg_mse_model) / avg_mse_mean) * 100
    
    print(f"\n   📌 MSE Comparison (on composed images, [0-255] scale):")
    print(f"      Model (Flow Matching):     {avg_mse_model:.2f}")
    print(f"      Baseline (Repeat Last):    {avg_mse_repeat:.2f}")
    print(f"      Baseline (Mean Frame):     {avg_mse_mean:.2f}")
    print(f"\n   🎯 Model Performance:")
    if improvement_vs_repeat > 0:
        print(f"      ✅ {improvement_vs_repeat:.1f}% better than repeating last frame")
    else:
        print(f"      ❌ {abs(improvement_vs_repeat):.1f}% worse than repeating last frame")
    
    if improvement_vs_mean > 0:
        print(f"      ✅ {improvement_vs_mean:.1f}% better than using mean frame")
    else:
        print(f"      ❌ {abs(improvement_vs_mean):.1f}% worse than using mean frame")
    
    # Overall assessment
    print(f"\n   💡 Assessment:")
    if improvement_vs_repeat > 20 and improvement_vs_mean > 10:
        print(f"      🌟 EXCELLENT: Model significantly outperforms baselines!")
    elif improvement_vs_repeat > 10:
        print(f"      ✅ GOOD: Model shows meaningful improvement over baselines")
    elif improvement_vs_repeat > 0:
        print(f"      ⚠️  MARGINAL: Model slightly better, but needs improvement")
    else:
        print(f"      ❌ POOR: Model doesn't beat simple baselines - needs more training")
    
    print(f"      Note: For dynamic smoke, even 10-20% improvement is significant!")
    
    print("\n" + "=" * 80)
    print("📝 SUMMARY:")
    print(f"   Condition: {num_cond} frames")
    print(f"   Generated: {num_gen} frames")
    print(f"   MSE (smoke only): {mse_model:.4f}")
    print(f"   MSE (visual): {avg_mse_model:.2f}")
    print(f"   Improvement vs baselines: {improvement_vs_repeat:.1f}% (repeat), {improvement_vs_mean:.1f}% (mean)")
    print(f"   Output: {args.output_grid}")
    print("=" * 80)


if __name__ == "__main__":
    main()
