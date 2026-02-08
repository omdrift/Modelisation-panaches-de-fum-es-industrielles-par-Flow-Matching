"""
Clean and filter dataset to remove bad smoke extractions.
Removes images with:
- Too few smoke pixels (isolated white points)
- Too many smoke pixels (entire image extracted)
- Poor smoke segmentation quality
"""
import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Clean smoke dataset")
    parser.add_argument("--input-dir", type=str, default="final_dataset", 
                       help="Input dataset directory")
    parser.add_argument("--min-smoke-ratio", type=float, default=0.01,
                       help="Minimum ratio of smoke pixels (default: 1%)")
    parser.add_argument("--max-smoke-ratio", type=float, default=0.90,
                       help="Maximum ratio of smoke pixels (default: 90%)")
    parser.add_argument("--min-brightness", type=float, default=0.1,
                       help="Minimum average brightness of smoke (0-1)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Only analyze, don't move/delete files")
    parser.add_argument("--mode", type=str, default="move", choices=["move", "delete"],
                       help="Move bad images to rejected/ or delete them")
    parser.add_argument("--show-examples", type=int, default=10,
                       help="Number of examples to show in report")
    return parser.parse_args()


def analyze_image(image_path):
    """
    Analyze smoke image quality.
    
    Returns:
        dict with statistics
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(float) / 255.0
        
        # Calculate metrics
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Smoke detection: pixels that are not completely black
        # In range [0, 1], black is < 0.05
        is_smoke = (img_array.max(axis=2) > 0.05)
        smoke_pixels = is_smoke.sum()
        smoke_ratio = smoke_pixels / total_pixels
        
        # Average brightness of smoke regions
        if smoke_pixels > 0:
            smoke_brightness = img_array[is_smoke].mean()
        else:
            smoke_brightness = 0.0
        
        # Check if image is mostly uniform (bad extraction)
        gray = img_array.mean(axis=2)
        variance = gray.var()
        
        return {
            'path': str(image_path),
            'total_pixels': total_pixels,
            'smoke_pixels': smoke_pixels,
            'smoke_ratio': smoke_ratio,
            'smoke_brightness': smoke_brightness,
            'variance': variance,
            'valid': True,
            'error': None
        }
    except Exception as e:
        return {
            'path': str(image_path),
            'valid': False,
            'error': str(e)
        }


def is_good_image(stats, min_ratio, max_ratio, min_brightness):
    """
    Determine if image meets quality criteria.
    """
    if not stats['valid']:
        return False, 'corrupted'
    
    # Check smoke ratio
    if stats['smoke_ratio'] < min_ratio:
        return False, f'too_few_pixels ({stats["smoke_ratio"]:.4f} < {min_ratio})'
    
    if stats['smoke_ratio'] > max_ratio:
        return False, f'too_many_pixels ({stats["smoke_ratio"]:.4f} > {max_ratio})'
    
    # Check brightness
    if stats['smoke_brightness'] < min_brightness:
        return False, f'too_dark ({stats["smoke_brightness"]:.4f} < {min_brightness})'
    
    # Check variance (too uniform = bad)
    if stats['variance'] < 0.001:
        return False, f'too_uniform (variance={stats["variance"]:.6f})'
    
    return True, 'good'


def process_split(split_dir, split_name, args, stats_summary):
    """Process one split (train/val/test)"""
    
    split_path = Path(split_dir) / split_name
    if not split_path.exists():
        print(f"Skipping {split_name}: directory not found")
        return
    
    rejected_dir = Path(split_dir) / f"rejected_{split_name}"
    if not args.dry_run and args.mode == "move":
        rejected_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(split_path.glob("*.png")) + list(split_path.glob("*.jpg"))
    
    print(f"\n{'='*60}")
    print(f"Processing {split_name}: {len(image_files)} images")
    print(f"{'='*60}")
    
    # Analyze all images
    good_images = []
    bad_images = []
    reasons = {}
    
    for img_path in tqdm(image_files, desc=f"Analyzing {split_name}"):
        stats = analyze_image(img_path)
        is_good, reason = is_good_image(
            stats, 
            args.min_smoke_ratio, 
            args.max_smoke_ratio,
            args.min_brightness
        )
        
        if is_good:
            good_images.append(stats)
        else:
            bad_images.append(stats)
            reasons[reason] = reasons.get(reason, 0) + 1
            
            # Move or delete bad image
            if not args.dry_run:
                if args.mode == "move":
                    dest = rejected_dir / img_path.name
                    shutil.move(str(img_path), str(dest))
                elif args.mode == "delete":
                    os.remove(img_path)
    
    # Print statistics
    print(f"\nResults for {split_name}:")
    print(f"  ✅ Good images: {len(good_images)} ({100*len(good_images)/len(image_files):.1f}%)")
    print(f"  ❌ Bad images:  {len(bad_images)} ({100*len(bad_images)/len(image_files):.1f}%)")
    
    if bad_images:
        print(f"\nReasons for rejection:")
        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {reason}: {count} images")
    
    # Show examples of bad images
    if bad_images and args.show_examples > 0:
        print(f"\nExamples of rejected images (first {args.show_examples}):")
        for i, stats in enumerate(bad_images[:args.show_examples]):
            _, reason = is_good_image(stats, args.min_smoke_ratio, args.max_smoke_ratio, args.min_brightness)
            print(f"  {i+1}. {Path(stats['path']).name}")
            print(f"     Reason: {reason}")
            print(f"     Smoke ratio: {stats['smoke_ratio']:.4f}, Brightness: {stats['smoke_brightness']:.4f}")
    
    # Save detailed statistics
    stats_summary[split_name] = {
        'total': len(image_files),
        'good': len(good_images),
        'bad': len(bad_images),
        'rejection_reasons': reasons,
        'good_examples': [s['path'] for s in good_images[:10]],
        'bad_examples': [s['path'] for s in bad_images[:10]]
    }
    
    return good_images, bad_images


def generate_report(stats_summary, output_path, args):
    """Generate HTML report with statistics"""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Cleaning Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .good { color: green; font-weight: bold; }
            .bad { color: red; font-weight: bold; }
            .params { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Dataset Cleaning Report</h1>
        
        <div class="params">
            <h2>Parameters</h2>
            <ul>
                <li><strong>Minimum smoke ratio:</strong> {:.1%}</li>
                <li><strong>Maximum smoke ratio:</strong> {:.1%}</li>
                <li><strong>Minimum brightness:</strong> {:.2f}</li>
                <li><strong>Mode:</strong> {}</li>
            </ul>
        </div>
    """.format(
        args.min_smoke_ratio,
        args.max_smoke_ratio,
        args.min_brightness,
        args.mode if not args.dry_run else "dry-run (no changes)"
    )
    
    # Summary table
    html += """
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Split</th>
                <th>Total Images</th>
                <th>Good Images</th>
                <th>Bad Images</th>
                <th>Success Rate</th>
            </tr>
    """
    
    total_all = 0
    good_all = 0
    bad_all = 0
    
    for split_name, stats in stats_summary.items():
        total = stats['total']
        good = stats['good']
        bad = stats['bad']
        rate = 100 * good / total if total > 0 else 0
        
        total_all += total
        good_all += good
        bad_all += bad
        
        html += f"""
            <tr>
                <td><strong>{split_name}</strong></td>
                <td>{total:,}</td>
                <td class="good">{good:,}</td>
                <td class="bad">{bad:,}</td>
                <td>{rate:.1f}%</td>
            </tr>
        """
    
    # Total row
    rate_all = 100 * good_all / total_all if total_all > 0 else 0
    html += f"""
            <tr style="background-color: #e8f5e9;">
                <td><strong>TOTAL</strong></td>
                <td><strong>{total_all:,}</strong></td>
                <td class="good"><strong>{good_all:,}</strong></td>
                <td class="bad"><strong>{bad_all:,}</strong></td>
                <td><strong>{rate_all:.1f}%</strong></td>
            </tr>
        </table>
    """
    
    # Rejection reasons
    html += "<h2>Rejection Reasons</h2>"
    
    for split_name, stats in stats_summary.items():
        if stats['rejection_reasons']:
            html += f"<h3>{split_name}</h3><ul>"
            for reason, count in sorted(stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
                html += f"<li><strong>{reason}:</strong> {count} images</li>"
            html += "</ul>"
    
    html += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n📄 HTML report saved to: {output_path}")


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        return
    
    print("🔍 Dataset Cleaning Tool")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Min smoke ratio: {args.min_smoke_ratio:.1%}")
    print(f"Max smoke ratio: {args.max_smoke_ratio:.1%}")
    print(f"Min brightness: {args.min_brightness:.2f}")
    print(f"Mode: {args.mode}")
    if args.dry_run:
        print("⚠️  DRY RUN MODE: No files will be modified")
    print("="*60)
    
    stats_summary = {}
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        process_split(input_dir, split_name, args, stats_summary)
    
    # Save statistics to JSON
    stats_file = input_dir / "cleaning_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"\n💾 Statistics saved to: {stats_file}")
    
    # Generate HTML report
    report_file = input_dir / "cleaning_report.html"
    generate_report(stats_summary, report_file, args)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total_all = sum(s['total'] for s in stats_summary.values())
    good_all = sum(s['good'] for s in stats_summary.values())
    bad_all = sum(s['bad'] for s in stats_summary.values())
    
    print(f"Total images processed: {total_all:,}")
    print(f"✅ Good images: {good_all:,} ({100*good_all/total_all:.1f}%)")
    print(f"❌ Bad images: {bad_all:,} ({100*bad_all/total_all:.1f}%)")
    
    if args.dry_run:
        print("\n⚠️  This was a dry run. No files were modified.")
        print("   Run without --dry-run to actually clean the dataset.")
    else:
        if args.mode == "move":
            print(f"\n📁 Bad images moved to: {input_dir}/rejected_*/ directories")
        elif args.mode == "delete":
            print(f"\n🗑️  Bad images deleted permanently")
    
    print("="*60)


if __name__ == "__main__":
    main()
