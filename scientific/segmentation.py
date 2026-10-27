"""Per-image CLIPSeg segmentation and spatial guided filtering, without matting."""
from __future__ import annotations

from importlib import metadata
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.nn import functional as F

from scientific.protocol import file_hash

MODEL_FILES = ('config.json', 'model.safetensors', 'preprocessor_config.json',
               'vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json')


def validate_settings(settings: dict) -> None:
    if settings.get('backend') != 'clipseg_guided':
        raise ValueError('The scientific extractor requires backend clipseg_guided')
    if not settings.get('model_id') or not re.fullmatch(r'[0-9a-f]{40}', settings.get('revision', '')):
        raise ValueError('CLIPSeg requires a model and immutable 40-character revision')
    if not re.fullmatch(r'[0-9a-f]{64}', settings.get('weights_sha256', '')):
        raise ValueError('CLIPSeg requires the expected safetensors SHA-256')
    if not isinstance(settings.get('negative_prompts'), list):
        raise ValueError('Negative prompts must be a list of strings')
    prompts = [settings.get('smoke_prompt'), *settings['negative_prompts']]
    if len(prompts) < 2 or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError('Require a smoke prompt and at least one negative prompt')
    low, high = settings['smoke_background_threshold'], settings['smoke_foreground_threshold']
    if not 0 <= low < high <= 1:
        raise ValueError('Require ordered smoke score thresholds in [0,1]')
    for key in ('negative_threshold', 'negative_margin'):
        if not math.isfinite(settings[key]) or not 0 <= settings[key] <= 1:
            raise ValueError('Negative score controls must be finite and in [0,1]')
    if type(settings['guided_radius']) is not int or settings['guided_radius'] < 1:
        raise ValueError('Guided-filter radius must be a positive integer')
    if not math.isfinite(settings['guided_epsilon']) or settings['guided_epsilon'] <= 0:
        raise ValueError('Guided-filter epsilon must be finite and positive')


def guided_filter(guide: np.ndarray, mask: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    """He et al. (ECCV 2010), grayscale guide, equations 5–8; no contrib package."""
    import cv2

    def mean(value):
        return cv2.boxFilter(value, -1, (2 * radius + 1, 2 * radius + 1),
                             normalize=True, borderType=cv2.BORDER_REFLECT_101)

    guide, mask = guide.astype(np.float32), mask.astype(np.float32)
    mean_i, mean_p = mean(guide), mean(mask)
    variance = np.maximum(mean(guide * guide) - mean_i * mean_i, 0)
    covariance = mean(guide * mask) - mean_i * mean_p
    a = covariance / (variance + epsilon)
    b = mean_p - a * mean_i
    return np.clip(mean(a) * guide + mean(b), 0, 1).astype(np.float32)


def refine_mask(frame: np.ndarray, scores: np.ndarray, settings: dict) -> np.ndarray:
    """Remap semantic scores then guide contours with this image's luminance."""
    import cv2

    expected = (1 + len(settings['negative_prompts']), *frame.shape[:2])
    if scores.shape != expected or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError('Invalid CLIPSeg scores: require finite per-prompt maps in [0,1]')
    smoke, negative = scores[0], scores[1:].max(axis=0)
    excluded = (negative > settings['negative_threshold']) & (negative > smoke + settings['negative_margin'])
    low, high = settings['smoke_background_threshold'], settings['smoke_foreground_threshold']
    mask = np.clip((smoke - low) / (high - low), 0, 1)
    mask[excluded] = 0
    guide = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    alpha = guided_filter(guide, mask, settings['guided_radius'], settings['guided_epsilon'])
    # Spatial smoothing must not restore explicitly excluded structures.
    alpha[excluded] = 0
    return alpha


class ClipSegExtractor:
    """Share each frame's visual encoding across prompts; frames stay independent."""

    def __init__(self, settings: dict, device='auto', cache_dir=None, local_files_only=False, batch_frames=None):
        validate_settings(settings)
        if batch_frames is not None and (type(batch_frames) is not int or batch_frames < 1):
            raise ValueError('batch_frames must be a positive integer')
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError('CLIPSeg requires transformers==4.49.0; update environment_scientific.yml') from exc
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested for CLIPSeg but unavailable')
        self.batch_frames = batch_frames if batch_frames is not None else 1
        self.settings = settings
        snapshot = Path(snapshot_download(settings['model_id'], revision=settings['revision'],
            allow_patterns=list(MODEL_FILES), cache_dir=cache_dir, local_files_only=local_files_only, token=False))
        fingerprints = {name: file_hash(snapshot / name) for name in MODEL_FILES}
        if fingerprints['model.safetensors'] != settings['weights_sha256']:
            raise ValueError('CLIPSeg weights SHA-256 differs from the frozen configuration')
        self.processor = CLIPSegProcessor.from_pretrained(snapshot, local_files_only=True, token=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained(snapshot, local_files_only=True,
            token=False, use_safetensors=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.prompts = [settings['smoke_prompt'], *settings['negative_prompts']]
        self.text_inputs = self.processor(text=self.prompts, padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            self.conditional_embeddings = self.model.get_conditional_embeddings(batch_size=len(self.prompts), **self.text_inputs)
        self.provenance = dict(backend=settings['backend'], model_id=settings['model_id'], revision=settings['revision'],
            files_sha256=fingerprints, implementation_sha256=file_hash(__file__), device=str(self.device),
            package_versions={name: metadata.version(name) for name in ('transformers', 'huggingface-hub', 'tokenizers', 'safetensors', 'torch', 'numpy', 'opencv-python-headless', 'Pillow')},
            mask_semantics='estimated_soft_semantic_mask_not_physical_opacity', temporal_smoothing=False,
            resize_before_segmentation=False, output_composition='source_rgb_times_mask_no_intensity_gain',
            inference=dict(engine='shared_vision_cached_text_fp32', batch_frames=self.batch_frames, cudnn_tf32=False))

    @torch.inference_mode()
    def predict_scores_batch(self, frames: np.ndarray) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(np.round(frame * 255).astype(np.uint8)) for frame in frames]
        inputs = self.processor(images=images, return_tensors='pt').to(self.device)
        # Same HF 4.49 forward path; visual activations do not depend on text.
        # Full FP32 convolutions avoid batch-dependent TF32 rounding at thresholds.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark, deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False):
            vision = self.model.clip.vision_model(pixel_values=inputs['pixel_values'], output_hidden_states=True,
                interpolate_pos_encoding=True, return_dict=True)
            prompts = len(self.prompts)
            activations = [vision.hidden_states[index + 1].repeat_interleave(prompts, dim=0) for index in self.model.extract_layers]
            conditions = self.conditional_embeddings.repeat(len(frames), 1)
            logits = self.model.decoder(activations, conditions, return_dict=True).logits
        if logits.ndim != 3 or logits.shape[0] != len(frames) * prompts or not torch.isfinite(logits).all():
            raise ValueError('Invalid CLIPSeg logits')
        # Match the prototype: resize logits at native resolution BEFORE sigmoid.
        logits = F.interpolate(logits[:, None], size=frames.shape[1:3], mode='bilinear', align_corners=False)[:, 0]
        return logits.sigmoid().reshape(len(frames), prompts, *frames.shape[1:3]).cpu().numpy()

    def predict_scores(self, frame: np.ndarray) -> np.ndarray:
        return self.predict_scores_batch(frame[None])[0]

    def __call__(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
            raise ValueError('Expected a nonempty RGB clip [T,H,W,3]')
        if not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 1:
            raise ValueError('RGB values must be finite and in [0,1]')
        masks = []
        for start in range(0, len(frames), self.batch_frames):
            batch = frames[start:start + self.batch_frames]
            try:
                scores = self.predict_scores_batch(batch)
            except ValueError as exc:
                raise ValueError(f'Segmentation failed at source frames {start}:{start + len(batch)}: {exc}') from exc
            for offset, frame in enumerate(batch):
                try:
                    masks.append(refine_mask(frame, scores[offset], self.settings))
                except ValueError as exc:
                    raise ValueError(f'Segmentation failed at source frame {start + offset}: {exc}') from exc
        alpha = np.stack(masks).astype(np.float32)
        if not np.isfinite(alpha).all():
            raise ValueError('Non-finite spatially refined mask')
        return (frames * alpha[..., None]).astype(np.float32), alpha
