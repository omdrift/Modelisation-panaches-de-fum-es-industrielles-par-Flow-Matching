import os
import random
from argparse import Namespace as ArgsNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset.text_based_video_dataset import TextBasedVideoDataset
from evaluation.evaluator import Evaluator
from lutils.configuration import Configuration
from lutils.distributed import setup_torch_distributed
from lutils.logger import Logger
from model.model import Model
from training.checkpoints import load_checkpoint as _load_checkpoint
from training.checkpoints import save_checkpoint as _save_checkpoint
from training.trainer import Trainer


def _resolve_split_file(data_root: str, split: str) -> str:
    candidates = [
        f"{split}_files.txt",
        f"{split}.txt",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(data_root, c)):
            return c
    raise FileNotFoundError(
        f"Could not find split file for '{split}' in {data_root}. Tried: {', '.join(candidates)}"
    )


def _build_dataset(config: Configuration, split: str, random_time: bool):
    data_cfg = config["data"]
    train_cfg = config["training"]
    if split == "train":
        # The current objective selects its target from the observed training window.
        frames_per_sample = int(train_cfg.get("num_observations", 10))
    else:
        eval_cfg = config["evaluation"]
        evaluation_window = int(eval_cfg.get("num_observations", 10)) + int(
            eval_cfg.get("frames_to_generate", 0)
        )
        frames_per_sample = max(int(data_cfg.get("frames_per_sample", 0)), evaluation_window)

    return TextBasedVideoDataset(
        data_path=data_cfg["data_root"],
        file_list=_resolve_split_file(data_cfg["data_root"], split),
        input_size=data_cfg["input_size"],
        crop_size=data_cfg["crop_size"],
        frames_per_sample=frames_per_sample,
        random_horizontal_flip=bool(data_cfg.get("random_horizontal_flip", False)) and split == "train",
        random_time=random_time,
        skip_short_videos=frames_per_sample > 1,
    )


def _set_seed(seed: int, rank: int):
    final_seed = int(seed) + int(rank)
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)


def train(rank: int, args: ArgsNamespace, temp_dir: str):
    config = Configuration(args.config)

    if args.num_gpus > 1:
        setup_torch_distributed(rank=rank, args=args, temp_dir=temp_dir)

    use_cuda = torch.cuda.is_available() and args.num_gpus > 0
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    _set_seed(args.random_seed, rank)

    train_dataset = _build_dataset(config, split="train", random_time=True)
    val_dataset = _build_dataset(config, split="val", random_time=False)

    model_config = config["model"]
    model_config["smoke_threshold"] = float(config["training"].get("smoke_threshold", 0.1))
    model = Model(model_config).to(device)
    if args.num_gpus > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    train_cfg = config["training"]
    optimizer_cfg = train_cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.module.vector_field_regressor.parameters() if isinstance(model, nn.parallel.DistributedDataParallel) else model.vector_field_regressor.parameters(),
        lr=optimizer_cfg["learning_rate"],
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )

    total_steps = int(optimizer_cfg["num_training_steps"])
    warmup_steps = int(optimizer_cfg.get("num_warmup_steps", 0))

    def _lr_lambda(step: int):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    logger = Logger(
        project="smoke-flow-matching",
        run_name=args.run_name,
        use_wandb=bool(args.wandb),
        config=config,
        rank=rank,
    )

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_step = 0
    if args.resume_step is not None:
        ckpt = run_dir / "checkpoints" / f"step_{args.resume_step}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        initial_step = _load_checkpoint(model, optimizer, scheduler, ckpt, device)
        logger.info(f"Resumed from {ckpt}")

    evaluator = Evaluator(
        rank=rank,
        config=config["evaluation"],
        dataset=val_dataset,
        device=device,
    )

    trainer = Trainer(
        rank=rank,
        world_size=max(1, int(args.num_gpus)),
        config=config["training"],
        dataset=train_dataset,
        device=device,
    )

    eval_every = int(train_cfg.get("eval_every", 1000))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 2000))

    def do_eval(step: int):
        evaluator.evaluate(
            model=model,
            logger=logger,
            global_step=step,
            max_num_batches=int(config["evaluation"].get("max_num_batches", 100)),
        )

    def do_save(step: int):
        if rank == 0:
            _save_checkpoint(model, optimizer, scheduler, run_dir, step)

    final_step = trainer.train_steps(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        initial_step=initial_step,
        total_steps=total_steps,
        eval_fn=do_eval,
        save_fn=do_save,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
    )

    if rank == 0:
        _save_checkpoint(model, optimizer, scheduler, run_dir, final_step)
        logger.info(f"Training finished at step {final_step}")

    if args.num_gpus > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
