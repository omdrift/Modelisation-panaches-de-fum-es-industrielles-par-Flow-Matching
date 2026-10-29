"""Complete fixed video panels with bounded local PNG/GIF retention."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw
import torch

from scientific.monitoring import preserve_rng
from scientific.protocol import write_json


def _rgb(images):
    images = torch.as_tensor(images).detach().cpu()
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError('Visual panels require finite [T,3,H,W] RGB images')
    return ((images.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()


def _name(window_id):
    value = str(window_id)
    stem = re.sub(r'[^A-Za-z0-9_.-]', '_', value)[:72]
    return f'{stem}_{hashlib.sha256(value.encode()).hexdigest()[:10]}'


def _tile(original, predicted, index, forecasting):
    height, width = original.shape[:2]
    # Magnitude on the same [0,255] scale; no image-specific contrast stretch.
    error = np.abs(original.astype(np.int16) - predicted.astype(np.int16)).astype(np.uint8)
    canvas = Image.new('RGB', (3 * width, height + 36), 'white')
    for column, image in enumerate((original, predicted, error)):
        canvas.paste(Image.fromarray(image), (column * width, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((3, 1), f'Future t+{index + 1}' if forecasting else f'Frame {index + 1}', fill='black')
    # For 64px images these labels fit each column; tiny test fixtures need larger tiles.
    for column, text in enumerate(('Original', 'Prediction' if forecasting else 'Reconstruction', 'Abs. error')):
        draw.text((column * width + 2, 18), text, fill='black')
    return canvas


def _context_strip(context, min_width):
    if context is None:
        return None
    images = _rgb(context)
    height, width = images.shape[1:3]
    strip = Image.new('RGB', (max(min_width, len(images) * width), height + 22), 'white')
    draw = ImageDraw.Draw(strip)
    draw.text((3, 2), f'OBSERVED CONTEXT: {len(images)} images', fill='black')
    for index, image in enumerate(images):
        strip.paste(Image.fromarray(image), (index * width, 22))
    return strip


def _render(sequence):
    originals, predictions = _rgb(sequence['target']), _rgb(sequence['prediction'])
    if originals.shape != predictions.shape or not len(originals):
        raise ValueError('Original and predicted visual sequences must be nonempty and aligned')
    # Enlarge tiny synthetic fixtures and use enough width for legible column labels.
    side = max(96, originals.shape[2])
    height = max(64, originals.shape[1])
    context = sequence.get('context')
    tiles = [_tile(np.asarray(Image.fromarray(a).resize((side, height))),
                   np.asarray(Image.fromarray(b).resize((side, height))), index, context is not None)
             for index, (a, b) in enumerate(zip(originals, predictions))]
    strip = _context_strip(context, tiles[0].width)
    columns = min(4, len(tiles))
    top = 0 if strip is None else strip.height
    grid = Image.new('RGB', (max(columns * tiles[0].width, 0 if strip is None else strip.width),
                             top + math.ceil(len(tiles) / columns) * tiles[0].height), 'white')
    if strip is not None:
        grid.paste(strip, (0, 0))
    frames = []
    for index, tile in enumerate(tiles):
        grid.paste(tile, ((index % columns) * tile.width, top + (index // columns) * tile.height))
        frame = Image.new('RGB', (max(tile.width, 0 if strip is None else strip.width), top + tile.height), 'white')
        if strip is not None:
            frame.paste(strip, (0, 0))
        frame.paste(tile, (0, top))
        frames.append(frame)
    return grid, frames


def _save_image(image, path, **kwargs):
    temporary = path.with_name(f'.{path.stem}.tmp{path.suffix}')
    try:
        image.save(temporary, **kwargs)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@preserve_rng()
def export_panel_media(run, step, method, snapshot, split, *, context_frames=5, best=False, final=False):
    """Write latest/best/final panels; returned paths can be logged to W&B.

    Only realization zero is displayed. Selection metrics still average all
    configured realizations. GIF frame duration is presentation timing, never
    a claim about the source acquisition interval.
    """
    if split not in ('train', 'train_eval', 'val', 'validation', 'test_dates', 'test_views'):
        raise ValueError('Unknown visual panel split')
    if split.startswith('test') and not final:
        raise ValueError('Test visualizations require frozen final selection')
    directory = Path(run) / 'media' / ('final' if final else 'latest') / split
    directory.mkdir(parents=True, exist_ok=True)
    examples, pngs, gifs = [], [], []
    for sequence in snapshot.sequences:
        context = sequence.get('context')
        if context is not None and len(context) != context_frames:
            raise ValueError('Visual context differs from the declared observations')
        grid, frames = _render(sequence)
        stem = _name(sequence['window_id'])
        png, gif = directory / f'{stem}.png', directory / f'{stem}.gif'
        _save_image(grid, png)
        _save_image(frames[0], gif, save_all=True, append_images=frames[1:], duration=200,
                    loop=0, optimize=False, disposal=2)
        pngs.append(png)
        gifs.append(gif)
        examples.append(dict(window_id=sequence['window_id'], png=png.name, gif=gif.name,
                             context_frames=0 if context is None else len(context),
                             future_frames=len(sequence['target']) if context is not None else 0,
                             reconstructed_frames=len(sequence['target']) if context is None else 0))
    expected = {path.name for path in pngs + gifs}
    for path in directory.iterdir():
        if path.suffix in ('.png', '.gif') and path.name not in expected:
            path.unlink()
    manifest = directory / 'manifest.json'
    write_json(manifest, dict(step=int(step), method=method, split=split, displayed_realization=0,
                              animation_interval_ms=200, animation_timing='presentation_only', examples=examples))
    if best:
        best_directory = Path(run) / 'media' / 'best' / split
        best_directory.mkdir(parents=True, exist_ok=True)
        for old in best_directory.iterdir():
            if old.is_file():
                old.unlink()
        for path in [*pngs, *gifs, manifest]:
            shutil.copy2(path, best_directory / path.name)
    return dict(manifest=manifest, pngs=pngs, gifs=gifs)
