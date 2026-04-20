"""
Training entry point for MagiHuman DiT model with LoRA / full fine-tuning.

Usage:
    # 1. First, pre-encode your training data into latents:
    python train.py --mode preprocess \
        --input-dir /path/to/raw_data \
        --output-dir /path/to/latent_data \
        --manifest /path/to/manifest.json \
        --engine_config.load /path/to/model_checkpoint \
        --evaluation_config.vae_model_path /path/to/vae \
        --evaluation_config.audio_model_path /path/to/audio_model \
        --evaluation_config.txt_model_path /path/to/t5_gemma

    # 2. Then, train with LoRA:
    torchrun --nproc_per_node=1 train.py --mode train \
        --latent-dir /path/to/latent_data \
        --engine_config.load /path/to/model_checkpoint \
        --lora-rank 16 \
        --lr 1e-4 \
        --num-epochs 10 \
        --output-dir checkpoints/lora_run1

    # 3. Or full fine-tune (requires much more memory):
    torchrun --nproc_per_node=8 train.py --mode train \
        --latent-dir /path/to/latent_data \
        --engine_config.load /path/to/model_checkpoint \
        --no-lora \
        --lr 1e-5 \
        --num-epochs 5

Manifest JSON format:
    [
        {
            "id": "sample_001",
            "video": "videos/clip1.mp4",
            "audio": "audio/clip1.wav",
            "prompt": "A person speaking with natural gestures...",
            "seconds": 5
        },
        ...
    ]
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="MagiHuman Training", allow_abbrev=False)

    parser.add_argument("--mode", type=str, required=True, choices=["train", "preprocess", "merge"],
                        help="Mode: 'train' for training, 'preprocess' for latent encoding, 'merge' for LoRA merging")

    # Data
    parser.add_argument("--latent-dir", type=str, default=None, help="Directory with pre-encoded .pt latent files")
    parser.add_argument("--input-dir", type=str, default=None, help="Raw data directory (for preprocess mode)")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest JSON file (for preprocess mode)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tune)")
    parser.add_argument("--lora-targets", type=str, nargs="+", default=None,
                        help="LoRA target module regex patterns")

    # Training
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "constant", "linear"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from")

    # Merge mode
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA checkpoint to merge (merge mode)")
    parser.add_argument("--merged-output", type=str, default=None, help="Output path for merged model")

    # Loss
    parser.add_argument("--video-loss-weight", type=float, default=1.0)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)

    args, remaining = parser.parse_known_args()
    return args, remaining


def setup_distributed():
    """Initialize distributed training if applicable."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)
        # Minimal single-GPU distributed setup
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl")
        return True
    return False


def run_preprocess(args, remaining_argv):
    """Pre-encode raw data into latents."""
    from inference.common.config import parse_config

    # Temporarily inject remaining args for pydantic config parsing
    sys.argv = [sys.argv[0]] + remaining_argv
    config = parse_config(verbose=True)

    from training.dataset import LatentPreprocessor
    preprocessor = LatentPreprocessor(config.evaluation_config, device="cuda")
    preprocessor.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    logger.info("Preprocessing complete!")


def run_merge(args, remaining_argv):
    """Merge LoRA weights into base model and save."""
    from inference.common.config import parse_config
    from inference.model.dit.dit_model import get_dit
    from training.lora import inject_lora, load_lora_weights, merge_lora_weights

    sys.argv = [sys.argv[0]] + remaining_argv
    config = parse_config(verbose=True)

    model = get_dit(config.arch_config, config.engine_config)

    # Determine LoRA config
    target_modules = args.lora_targets or [
        r"block\.layers\.\d+\.attention\.linear_qkv",
        r"block\.layers\.\d+\.attention\.linear_proj",
    ]
    inject_lora(model, target_modules, rank=args.lora_rank, alpha=args.lora_alpha)
    load_lora_weights(model, args.lora_path)
    merge_lora_weights(model)

    output_path = args.merged_output or "merged_model.pt"
    torch.save(model.state_dict(), output_path)
    logger.info(f"Merged model saved to {output_path}")


def run_train(args, remaining_argv):
    """Run training."""
    from inference.common.config import parse_config
    from inference.model.dit.dit_model import get_dit
    from training.trainer import MagiTrainer, TrainerConfig
    from training.dataset import MagiTrainingDataset

    setup_distributed()

    # Parse model config via pydantic
    sys.argv = [sys.argv[0]] + remaining_argv
    config = parse_config(verbose=True)

    # Build model
    model = get_dit(config.arch_config, config.engine_config)

    # Build trainer config
    lora_targets = args.lora_targets or [
        r"block\.layers\.\d+\.attention\.linear_qkv",
        r"block\.layers\.\d+\.attention\.linear_proj",
    ]
    trainer_config = TrainerConfig(
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        output_dir=args.output_dir,
        log_every=args.log_every,
        save_every=args.save_every,
        video_loss_weight=args.video_loss_weight,
        audio_loss_weight=args.audio_loss_weight,
        patch_size=config.evaluation_config.data_proxy_config.patch_size,
        t_patch_size=config.evaluation_config.data_proxy_config.t_patch_size,
        coords_style=config.evaluation_config.data_proxy_config.coords_style,
    )

    # Build trainer
    trainer = MagiTrainer(
        model=model,
        trainer_config=trainer_config,
        model_config=config.arch_config,
    )

    # Build dataset and dataloader
    dataset = MagiTrainingDataset(
        latent_dir=args.latent_dir,
        patch_size=trainer_config.patch_size,
        t_patch_size=trainer_config.t_patch_size,
        coords_style=trainer_config.coords_style,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Train
    trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        resume_from=args.resume_from,
    )


def main():
    args, remaining = parse_args()

    if args.mode == "preprocess":
        if not args.input_dir or not args.manifest:
            logger.error("--input-dir and --manifest are required for preprocess mode")
            sys.exit(1)
        run_preprocess(args, remaining)

    elif args.mode == "merge":
        if not args.lora_path:
            logger.error("--lora-path is required for merge mode")
            sys.exit(1)
        run_merge(args, remaining)

    elif args.mode == "train":
        if not args.latent_dir:
            logger.error("--latent-dir is required for train mode")
            sys.exit(1)
        run_train(args, remaining)


if __name__ == "__main__":
    main()
