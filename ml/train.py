"""
Training script for DualBranchPhaseNet.
Optimized for batch execution on cluster nodes.
Uses folder-isolated checkpoint paths (`checkpoint_dir/run_name/`) and accepts
direct complex phase maps (ab_in, ab_out) from RMDataset.
"""

import argparse
import datetime
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# CLI Configuration
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DualBranchPhaseNet on IITM Aqua.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Parameters
    p.add_argument("--N", type=int, default=32, help="Grid spatial resolution N x N.")
    p.add_argument(
        "--zern_n", type=int, default=5, help="Maximum Zernike radial order."
    )
    p.add_argument("--epochs", type=int, default=25, help="Total training epochs.")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")

    # Checkpoint & Path Options
    p.add_argument(
        "--load_ckpt_path",
        type=str,
        default=None,
        help="Explicit path to a .pth file to resume from or load weights from.",
    )
    p.add_argument(
        "--weights_only",
        action="store_true",
        help="Restore only model weights (ignore optimizer/scheduler/epoch state).",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint",
        help="Base directory for writing subfolder-isolated checkpoints and plots.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to cache dataset .pt files.",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="runs",
        help="Base directory for TensorBoard logs.",
    )
    p.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing local model dependencies.",
    )

    # Dataset Sizes
    p.add_argument("--train_size", type=int, default=4096, help="Training set size.")
    p.add_argument("--val_size", type=int, default=256, help="Validation set size.")

    # Architecture & Scheduler Tuning
    p.add_argument(
        "--embed_dim", type=int, default=256, help="Net embedding dimension."
    )
    p.add_argument(
        "--eta_min", type=float, default=1e-6, help="Minimum learning rate floor."
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "warmup_cosine"],
        help="Learning rate schedule algorithm.",
    )
    p.add_argument(
        "--warmup_epochs", type=int, default=5, help="Epochs for linear warmup."
    )
    p.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Initial LR fraction for warmup.",
    )
    p.add_argument(
        "--plateau_factor",
        type=float,
        default=0.5,
        help="LR reduction factor on plateau.",
    )
    p.add_argument(
        "--plateau_patience",
        type=int,
        default=5,
        help="Patience epochs before reducing LR.",
    )

    # Loss Term Weights
    p.add_argument(
        "--alpha_phase", type=float, default=1.0, help="Phase loss term weight."
    )
    p.add_argument(
        "--alpha_kernel", type=float, default=0.0, help="Kernel loss term weight."
    )

    # Rk Noise Injection (sim-to-real robustness)
    p.add_argument(
        "--train_noise",
        action="store_true",
        help="Enable random complex AWGN injection on Rk during training, "
        "with a fresh per-sample SNR drawn each batch from "
        "[--snr_min, --snr_max] (plus a --clean_prob chance of no noise). "
        "Off by default so existing invocations are unaffected.",
    )
    p.add_argument(
        "--snr_min",
        type=float,
        default=0.0,
        help="Minimum SNR (dB) for random noise injection during training "
        "(only used when --train_noise is set).",
    )
    p.add_argument(
        "--snr_max",
        type=float,
        default=30.0,
        help="Maximum SNR (dB) for random noise injection during training "
        "(only used when --train_noise is set).",
    )
    p.add_argument(
        "--clean_prob",
        type=float,
        default=0.1,
        help="Probability a training sample is left noise-free (SNR=inf) "
        "even when --train_noise is set, so the model doesn't lose its "
        "ability to exploit high-SNR signal.",
    )
    p.add_argument(
        "--noise_warmup_epochs",
        type=int,
        default=0,
        help="Number of initial epochs trained noise-free before "
        "--train_noise kicks in (curriculum warmup).",
    )
    p.add_argument(
        "--val_snr_bins",
        type=str,
        default="20,10,0",
        help="Comma-separated extra SNR values (dB) for periodic noisy "
        "validation tracking, logged separately from the main (clean) "
        "val loss used for checkpointing/scheduling. 'inf' is accepted. "
        "Pass an empty string to disable.",
    )
    p.add_argument(
        "--val_snr_every",
        type=int,
        default=5,
        help="Run the extra noisy-validation SNR bins every N epochs.",
    )

    # Environment & Execution Controls
    p.add_argument("--seed", type=int, default=42, help="Training random seed.")
    p.add_argument("--val_seed", type=int, default=420, help="Validation random seed.")
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save numbered checkpoint every N epochs.",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Update 'latest.pth' checkpoint every N epochs.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader worker count (-1 auto-detects from CPU/cluster environment).",
    )
    p.add_argument(
        "--no_shuffle_train",
        action="store_true",
        help="Disable training data loader shuffling.",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="TensorBoard & Checkpoint run identifier.",
    )

    return p.parse_args()


def resolve_num_workers(requested: int) -> int:
    """Detect appropriate CPU worker count from environment scheduler hints if requested == -1."""
    if requested >= 0:
        return requested
    for env_var in ("PBS_NUM_PPN", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = os.environ.get(env_var)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


def parse_snr_bins(raw: str) -> List[float]:
    """Parses a comma-separated SNR list (dB), accepting 'inf'. Empty string -> []."""
    raw = raw.strip()
    if not raw:
        return []
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


# --------------------------------------------------------------------------- #
# Logging Setup
# --------------------------------------------------------------------------- #
logger = logging.getLogger("dual_branch_phasenet_train")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_file_logging(log_path: Path) -> None:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for handler in logger.handlers:
        handler.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


# --------------------------------------------------------------------------- #
# Checkpoint Infrastructure
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path: Path,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: Dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[int, Optional[float], float, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    val_loss = checkpoint.get("val_loss")
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    config = checkpoint.get("config", {})
    return epoch, val_loss, best_val_loss, config


# --------------------------------------------------------------------------- #
# Learning Rate Scheduler Factory
# --------------------------------------------------------------------------- #
def build_scheduler(
    optimizer: torch.optim.Optimizer, args: argparse.Namespace
) -> Tuple[Any, bool]:
    if args.scheduler == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.eta_min
            ),
            False,
        )

    if args.scheduler == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.plateau_factor,
                patience=args.plateau_patience,
                min_lr=args.eta_min,
            ),
            True,
        )

    if args.scheduler == "warmup_cosine":
        warmup_epochs = max(0, min(args.warmup_epochs, args.epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=args.eta_min,
        )
        return (
            torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            ),
            False,
        )

    raise ValueError(f"Unknown --scheduler: {args.scheduler!r}")


# --------------------------------------------------------------------------- #
# Image & Signal Processing Helpers
# --------------------------------------------------------------------------- #
def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top : top + size, left : left + size]


def compute_object_kernel(obj: torch.Tensor) -> torch.Tensor:
    obj_c = obj if torch.is_complex(obj) else obj.to(torch.complex64)
    shifted = torch.fft.fftshift(obj_c, dim=(-2, -1))
    Ok = torch.fft.fft2(shifted, dim=(-2, -1)) / 4.0
    Ok = torch.fft.ifftshift(Ok, dim=(-2, -1))
    return Ok[..., 1:, 1:]


def align_global_phase(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inner_prod = torch.sum(pred * torch.conj(target), dim=(-2, -1), keepdim=True)
    return pred * torch.exp(-1j * torch.angle(inner_prod))


def phase_error_stats(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Tuple[float, float, torch.Tensor]:
    diff = torch.angle(pred * torch.conj(target)).abs()
    valid = mask.unsqueeze(0).expand_as(diff) > 0.5
    return diff[valid].mean().item(), diff[valid].max().item(), diff


def plot_comparison(
    pred: torch.Tensor,
    target: torch.Tensor,
    diff: torch.Tensor,
    mask: torch.Tensor,
    title: str,
    out_path: Path,
    idx: int = 0,
) -> None:
    p, t, m, d = pred[idx].cpu(), target[idx].cpu(), mask.cpu(), diff[idx].cpu()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    panels = [
        (t.angle(), f"{title} Target Phase"),
        (p.angle(), f"{title} Pred Phase (Aligned)"),
        (d, f"{title} Angle Difference"),
    ]

    for ax, (data, panel_title) in zip(axes, panels):
        masked_data = torch.where(m > 0.5, data, torch.nan)
        ax.imshow(masked_data, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax.set_title(panel_title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Rk Noise Injection Helpers
# --------------------------------------------------------------------------- #
def sample_snr_db(
    batch_size: int,
    snr_min: float,
    snr_max: float,
    clean_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """Draws one SNR (dB) per sample, uniform in [snr_min, snr_max], with
    `clean_prob` probability of being float('inf') (i.e. noise-free)."""
    snr = torch.empty(batch_size, 1, 1, device=device).uniform_(snr_min, snr_max)
    if clean_prob > 0:
        clean_mask = torch.rand(batch_size, 1, 1, device=device) < clean_prob
        snr = torch.where(clean_mask, torch.full_like(snr, float("inf")), snr)
    return snr


def add_complex_noise(Rk: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Adds complex AWGN to Rk to match a per-sample target SNR (dB).

    snr_db is broadcastable against Rk's leading batch dim, e.g. shape
    (B, 1, 1). Entries equal to float('inf') naturally add zero noise
    (10**(inf/10) == inf -> noise_power == 0), so no special-casing needed.
    """
    sig_power = torch.mean(torch.abs(Rk) ** 2, dim=(-2, -1), keepdim=True)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = torch.sqrt(noise_power / 2.0)
    noise_real = torch.randn_like(Rk.real) * noise_std
    noise_imag = torch.randn_like(Rk.imag) * noise_std
    return Rk + torch.complex(noise_real, noise_imag)


# --------------------------------------------------------------------------- #
# Main Execution Pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    val_snr_bins = parse_snr_bins(args.val_snr_bins)

    # Dynamic package import path injection
    sys.path.insert(0, args.repo_root)
    from data_gen import RMDataset
    from dual_branch_phasenet import DualBranchPhaseNet, PhaseRetrievalLoss
    from forward import Simulation
    from rm import get_Rk_batched

    # Strategy 1 Subfolder Setup
    run_name = (
        args.run_name or f"DualBranchPhaseNet_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )

    base_checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    run_ckpt_dir = base_checkpoint_dir / run_name
    cache_dir = Path(args.cache_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser()

    for directory in (run_ckpt_dir, cache_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    latest_ckpt_path = run_ckpt_dir / "latest.pth"
    best_ckpt_path = run_ckpt_dir / "best.pth"

    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(run_dir / "train.log")

    logger.info(f"Run name: {run_name}")
    logger.info(f"Checkpoints directory: {run_ckpt_dir}")
    logger.info(f"Logging to console and: {run_dir / 'train.log'}")
    if args.train_noise:
        logger.info(
            f"Rk noise injection: ENABLED | SNR range=[{args.snr_min}, {args.snr_max}] dB "
            f"| clean_prob={args.clean_prob} | warmup_epochs={args.noise_warmup_epochs}"
        )
    else:
        logger.info("Rk noise injection: disabled")
    if val_snr_bins:
        logger.info(
            f"Noisy validation tracking: bins={val_snr_bins} dB, every {args.val_snr_every} epoch(s)"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    num_workers = resolve_num_workers(args.num_workers)
    logger.info(f"DataLoader workers: {num_workers}")

    N = args.N

    # Dataset & Dataloader setup
    train_dataset = RMDataset(
        N=2 * N,
        size=args.train_size,
        zern_n=args.zern_n,
        seed=args.seed,
        cache_path=str(cache_dir / f"train_N{N}_size{args.train_size}.pt"),
        num_workers_build=num_workers,
    )
    val_dataset = RMDataset(
        N=2 * N,
        size=args.val_size,
        zern_n=args.zern_n,
        seed=args.val_seed,
        cache_path=str(cache_dir / f"val_N{N}_size{args.val_size}.pt"),
        num_workers_build=num_workers,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Physics Simulation Setup
    simulation = Simulation(N, dtype=torch.complex64).to(device)

    # Derive aperture mask directly from dataset output phase map
    with torch.no_grad():
        ab_in_sample, _, _ = next(iter(train_loader))
        ab_sample_cropped = center_crop(ab_in_sample.to(device), N)
        aperture_mask = (ab_sample_cropped[0].abs() > 1e-6).float()

    model = DualBranchPhaseNet(
        N=N, embed_dim=args.embed_dim, aperture_mask=aperture_mask
    ).to(device)
    logger.info(f"Model param count: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler, scheduler_needs_metric = build_scheduler(optimizer, args)
    logger.info(f"Scheduler: {args.scheduler}")

    config = {
        "N": N,
        "zern_n": args.zern_n,
        "embed_dim": args.embed_dim,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "scheduler": args.scheduler,
        "train_noise": args.train_noise,
        "snr_min": args.snr_min if args.train_noise else None,
        "snr_max": args.snr_max if args.train_noise else None,
        "clean_prob": args.clean_prob if args.train_noise else None,
        "noise_warmup_epochs": args.noise_warmup_epochs if args.train_noise else None,
    }

    start_epoch = 0
    best_val_loss = float("inf")

    # Checkpoint Restoration Logic
    if args.load_ckpt_path:
        ckpt_path = Path(args.load_ckpt_path).expanduser()
        if ckpt_path.exists():
            if args.weights_only:
                load_checkpoint(ckpt_path, model, map_location=device)
                logger.info(
                    f"Loaded weights only from '{ckpt_path}'. Starting at epoch 0 with fresh optimizer/scheduler."
                )
            else:
                start_epoch, _, best_val_loss, _ = load_checkpoint(
                    ckpt_path, model, optimizer, scheduler, map_location=device
                )
                start_epoch += 1
                logger.info(
                    f"Resumed training from '{ckpt_path}' at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}."
                )
        else:
            logger.warning(
                f"Specified checkpoint '{ckpt_path}' non-existent. Training from scratch."
            )
    else:
        logger.info("No checkpoint path provided. Training from scratch.")

    criterion = PhaseRetrievalLoss(
        alpha_phase=args.alpha_phase, alpha_kernel=args.alpha_kernel
    )
    writer = SummaryWriter(str(log_dir / run_name))

    # TensorBoard Hyperparameter Logging Setup
    hparams = {k: v for k, v in vars(args).items() if v is not None}
    writer.add_text(
        "config", "\n".join(f"- **{k}**: {v}" for k, v in sorted(hparams.items()))
    )

    # Execution Loop Helper
    def run_epoch(
        dataloader: DataLoader,
        is_train: bool,
        epoch: int,
        inject_noise: bool = False,
    ) -> float:
        model.train() if is_train else model.eval()
        total_loss = 0.0

        with torch.set_grad_enabled(is_train):
            for ab_in, ab_out, obj in dataloader:
                ab_in, ab_out, obj = ab_in.to(device), ab_out.to(device), obj.to(device)

                with torch.no_grad():
                    k_outs = simulation(ab_in, ab_out, obj)
                    Rk = get_Rk_batched(
                        k_in=simulation.k_in_cropped, k_outs=k_outs, N=N
                    )
                    if inject_noise:
                        snr_db = sample_snr_db(
                            Rk.shape[0],
                            args.snr_min,
                            args.snr_max,
                            args.clean_prob,
                            device,
                        )
                        Rk = add_complex_noise(Rk, snr_db)
                    p_target, q_target = center_crop(ab_in, N), center_crop(ab_out, N)
                    o_target = compute_object_kernel(obj)

                if is_train:
                    optimizer.zero_grad()

                p_pred, q_pred, o_pred = model(Rk)
                loss = criterion(p_pred, q_pred, o_pred, p_target, q_target, o_target)

                if is_train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        phase_label = "train" if is_train else "val"
        writer.add_scalar(f"Loss/{phase_label}", avg_loss, epoch)
        return avg_loss

    # Fixed-SNR noisy validation, tracked separately from the main (clean)
    # val loss so checkpoint selection / plateau scheduling stay unaffected.
    def evaluate_at_snr(dataloader: DataLoader, snr_db_value: float) -> float:
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for ab_in, ab_out, obj in dataloader:
                ab_in, ab_out, obj = ab_in.to(device), ab_out.to(device), obj.to(device)

                k_outs = simulation(ab_in, ab_out, obj)
                Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)
                snr_db = torch.full(
                    (Rk.shape[0], 1, 1),
                    snr_db_value,
                    device=device,
                    dtype=torch.float32,
                )
                Rk = add_complex_noise(Rk, snr_db)

                p_target, q_target = center_crop(ab_in, N), center_crop(ab_out, N)
                o_target = compute_object_kernel(obj)

                p_pred, q_pred, o_pred = model(Rk)
                loss = criterion(p_pred, q_pred, o_pred, p_target, q_target, o_target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    # Main Training Cycle
    logger.info("--- Starting Training ---")
    train_loss = val_loss = float("nan")

    for epoch in tqdm(range(start_epoch, args.epochs)):
        writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], epoch)

        noise_active = args.train_noise and epoch >= args.noise_warmup_epochs
        train_loss = run_epoch(
            train_loader, is_train=True, epoch=epoch, inject_noise=noise_active
        )
        val_loss = run_epoch(val_loader, is_train=False, epoch=epoch)

        scheduler.step(val_loss) if scheduler_needs_metric else scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            + (" | noise: ON" if noise_active else "")
        )

        # Periodic noisy-validation SNR bins (diagnostic only)
        if val_snr_bins and (
            (epoch + 1) % args.val_snr_every == 0 or epoch == args.epochs - 1
        ):
            for snr in val_snr_bins:
                snr_val_loss = evaluate_at_snr(val_loader, snr)
                label = "clean" if math_isinf(snr) else f"{snr:g}dB"
                writer.add_scalar(f"Loss/val_snr_{label}", snr_val_loss, epoch)
                logger.info(f"  Noisy val @ {label}: {snr_val_loss:.4f}")

        writer.flush()

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_ckpt_path,
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

        # Save Periodic Checkpoint
        if (epoch + 1) % args.checkpoint_every == 0:
            periodic_path = run_ckpt_dir / f"epoch_{epoch + 1:04d}.pth"
            save_checkpoint(
                periodic_path,
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

        # Save Latest Model
        if (epoch + 1) % args.save_every == 0 or (epoch == args.epochs - 1):
            save_checkpoint(
                latest_ckpt_path,
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

    # TensorBoard Final Metrics
    writer.add_hparams(
        hparams,
        {
            "hparam/best_val_loss": best_val_loss,
            "hparam/final_train_loss": train_loss,
            "hparam/final_val_loss": val_loss,
        },
        run_name=".",
    )
    writer.close()
    logger.info(f"--- Training complete. Best val loss: {best_val_loss:.4f} ---")

    # Evaluation Plotting on Best Model Weights
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, map_location=device)
    model.eval()

    with torch.no_grad():
        ab_in, ab_out, obj = next(iter(val_loader))
        ab_in, ab_out, obj = ab_in.to(device), ab_out.to(device), obj.to(device)

        k_outs = simulation(ab_in, ab_out, obj)
        Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)
        p_target, q_target = center_crop(ab_in, N), center_crop(ab_out, N)

        p_pred, q_pred, _ = model(Rk)
        p_pred = align_global_phase(p_pred, p_target)
        q_pred = align_global_phase(q_pred, q_target)

    p_mean_err, p_max_err, p_diff = phase_error_stats(p_pred, p_target, aperture_mask)
    q_mean_err, q_max_err, q_diff = phase_error_stats(q_pred, q_target, aperture_mask)

    logger.info(
        f"p: mean |phase error| = {p_mean_err:.4f} rad, max = {p_max_err:.4f} rad"
    )
    logger.info(
        f"q: mean |phase error| = {q_mean_err:.4f} rad, max = {q_max_err:.4f} rad"
    )

    plot_comparison(
        p_pred,
        p_target,
        p_diff,
        aperture_mask,
        "p",
        run_ckpt_dir / "eval_p_comparison.png",
    )
    plot_comparison(
        q_pred,
        q_target,
        q_diff,
        aperture_mask,
        "q",
        run_ckpt_dir / "eval_q_comparison.png",
    )
    logger.info(f"Saved evaluation plots to {run_ckpt_dir}")


def math_isinf(x: float) -> bool:
    return x == float("inf") or x == float("-inf")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error(
            "Training crashed with an unhandled exception:\n%s", traceback.format_exc()
        )
        raise
