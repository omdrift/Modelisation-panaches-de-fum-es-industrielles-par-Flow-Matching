"""Compare original HF image/prompt inference with shared visual encodings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scientific.protocol import file_hash, load_protocol, write_json
from scientific.segmentation import ClipSegExtractor, refine_mask


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-npz', required=True, help='Contains native RGB uint8 raw_frames')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--cache-dir', default='scientific_models/huggingface')
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8])
    args = parser.parse_args(argv)
    if min(args.frames, args.repeats, *args.batch_sizes) < 1:
        parser.error('Frame count, repeats and batch sizes must be positive')
    with np.load(args.input_npz, allow_pickle=False) as data:
        raw = data['raw_frames']
    if raw.ndim != 4 or raw.shape[-1] != 3 or raw.dtype != np.uint8 or args.frames > len(raw):
        parser.error('Require uint8 RGB raw_frames and enough source images')
    indices = np.linspace(0, len(raw) - 1, args.frames, dtype=int)
    frames = raw[indices].astype(np.float32) / 255
    cfg = load_protocol()

    @torch.inference_mode()
    def reference():
        from PIL import Image
        result = []
        for frame in frames:
            image = Image.fromarray(np.round(frame * 255).astype(np.uint8))
            inputs = extractor.processor(images=[image] * len(extractor.prompts), return_tensors='pt').to(extractor.device)
            inputs.update(extractor.text_inputs)
            logits = extractor.model(**inputs).logits
            logits = F.interpolate(logits[:, None], size=frame.shape[:2], mode='bilinear', align_corners=False)[:, 0]
            result.append(logits.sigmoid().cpu().numpy())
        return np.stack(result)

    def optimized(batch):
        return np.concatenate([extractor.predict_scores_batch(frames[start:start + batch]) for start in range(0, len(frames), batch)])

    def synchronize():
        if extractor.device.type == 'cuda':
            torch.cuda.synchronize(extractor.device)

    results = []
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        extractor = ClipSegExtractor(cfg['preprocessing'], device=args.device, cache_dir=args.cache_dir, local_files_only=True)
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled, benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic, allow_tf32=False):
            variants = [('baseline', reference)] + [(f'shared_vision_batch_{batch}', lambda batch=batch: optimized(batch)) for batch in args.batch_sizes]
            for name, operation in variants:
                operation()  # Warm up CUDA kernels before measuring.
                times = []
                for _ in range(args.repeats):
                    synchronize()
                    if extractor.device.type == 'cuda':
                        torch.cuda.reset_peak_memory_stats(extractor.device)
                    start = time.perf_counter()
                    scores = operation()
                    synchronize()
                    times.append(time.perf_counter() - start)
                masks = np.stack([refine_mask(frame, score, cfg['preprocessing']) for frame, score in zip(frames, scores)])
                if name == 'baseline':
                    baseline_scores, baseline_masks = scores.copy(), masks.copy()
                np.testing.assert_allclose(scores, baseline_scores, rtol=1e-4, atol=1e-5)
                np.testing.assert_allclose(masks, baseline_masks, rtol=1e-4, atol=1e-4)
                row = dict(variant=name, frames=len(frames), seconds_median=float(np.median(times)), seconds_repeats=times,
                    frames_per_second=len(frames) / float(np.median(times)),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(extractor.device) if extractor.device.type == 'cuda' else None,
                    max_score_abs_difference=float(np.max(np.abs(scores - baseline_scores))),
                    max_mask_abs_difference=float(np.max(np.abs(masks - baseline_masks))))
                results.append(row)
                print(json.dumps(row), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
    write_json(args.output, dict(gpu=torch.cuda.get_device_name(extractor.device) if extractor.device.type == 'cuda' else None,
        torch=str(torch.__version__), cuda=torch.version.cuda, cpu_threads=torch.get_num_threads(), cudnn_tf32=False, matmul_tf32=False,
        source_npz_sha256=file_hash(args.input_npz), frame_indices=indices.tolist(), input_shape=list(frames.shape),
        benchmark_sha256=file_hash(__file__), extraction=extractor.provenance, results=results,
        metric='inference only, excludes loading, guided filtering, decode and file compression'))


if __name__ == '__main__':
    main()
