"""
Training script for DualBranchPhaseNet, adapted from the Colab notebook
(dual_branch_phasenet.ipynb) for batch execution on IITM Aqua.

Assumes this file lives alongside the malmo/ml package modules
(data_gen.py, zern.py, forward.py, rm.py, dual_branch_phasenet.py) --
i.e. it replaces the notebook inside the `malmo/ml` directory. If you keep
it elsewhere, point --repo_root at the malmo/ml directory.

Example (train from scratch with a cosine schedule, initial lr 4e-3):

    python train.py \
        --N 32 --zern_n 4 --epochs 25 --batch_size 32 \
        --lr 4e-3 --load_ckpt none \
        --train_size 4096 --val_size 256 \
        --checkpoint_dir ~/malmo/ml/checkpoint \
        --cache_dir ~/malmo/ml/cache \
        --log_dir ~/malmo/ml/runs
"""

import argparse
import datetime
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # no display on a compute node
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Train DualBranchPhaseNet on IITM Aqua.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- core interface, as specified ---
    p.add_argument("--N", type=int, default=32, help="Grid spatial resolution N x N.")
    p.add_argument(
        "--zern_n", type=int, default=4, help="Maximum Zernike polynomial radial order."
    )
    p.add_argument("--epochs", type=int, default=25, help="Total training epochs.")
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation loaders.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for AdamW optimizer.",
    )
    p.add_argument(
        "--load_ckpt",
        type=str,
        default="none",
        choices=["none", "latest", "best"],
        help="Checkpoint initialization strategy.",
    )
    p.add_argument(
        "--weights_only",
        action="store_true",
        help=(
            "When --load_ckpt is 'latest' or 'best', restore only the model "
            "weights from that checkpoint. Optimizer state, scheduler state, "
            "epoch, and best_val_loss are ignored -- training starts at "
            "epoch 0 with a fresh optimizer/scheduler and best_val_loss=inf. "
            "Has no effect when --load_ckpt=none."
        ),
    )
    p.add_argument(
        "--init_from_zern",
        type=int,
        default=None,
        help=(
            "zern_n whose checkpoint (under --load_ckpt) to load from -- lets "
            "you warm-start a run at --zern_n from a checkpoint trained at a "
            "different (typically lower) zern_n. Defaults to --zern_n, i.e. "
            "load from the same zern_n's checkpoint lineage."
        ),
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="~/malmo/ml/checkpoint",
        help="Directory to read/write .pth checkpoints and plot outputs.",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="~/malmo/ml/cache",
        help="Directory to cache generated dataset .pt files.",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="~/malmo/ml/runs",
        help="Base directory for TensorBoard logging.",
    )

    # --- dataset sizes ---
    p.add_argument(
        "--train_size",
        type=int,
        default=4096,
        help="Number of samples in the training dataset.",
    )
    p.add_argument(
        "--val_size",
        type=int,
        default=256,
        help="Number of samples in the validation dataset.",
    )

    # --- secondary knobs (sane defaults matching the notebook; safe to leave alone) ---
    p.add_argument(
        "--embed_dim",
        type=int,
        default=256,
        help="DualBranchPhaseNet embedding dimension.",
    )
    p.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="Minimum LR: floor for cosine schedules, min_lr for ReduceLROnPlateau.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau", "warmup_cosine"],
        help=(
            "LR schedule. 'cosine': CosineAnnealingLR over the full run. "
            "'plateau': ReduceLROnPlateau, stepped on val loss each epoch. "
            "'warmup_cosine': linear warmup for --warmup_epochs, then cosine "
            "annealing for the remaining epochs."
        ),
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Epochs of linear LR warmup. Only used when --scheduler=warmup_cosine.",
    )
    p.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help=(
            "LR at the start of warmup, as a fraction of --lr. Only used when "
            "--scheduler=warmup_cosine."
        ),
    )
    p.add_argument(
        "--plateau_factor",
        type=float,
        default=0.5,
        help="LR multiplier applied on plateau. Only used when --scheduler=plateau.",
    )
    p.add_argument(
        "--plateau_patience",
        type=int,
        default=5,
        help=(
            "Epochs with no val-loss improvement before reducing LR. Only used "
            "when --scheduler=plateau."
        ),
    )
    p.add_argument(
        "--alpha_phase",
        type=float,
        default=1.0,
        help="Phase term weight in PhaseRetrievalLoss.",
    )
    p.add_argument(
        "--alpha_kernel",
        type=float,
        default=0.0,
        help="Kernel term weight in PhaseRetrievalLoss.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--val_seed", type=int, default=420, help="Random seed for the validation set."
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=25,
        help="Save a numbered checkpoint every N epochs.",
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=1,
        help=(
            "Write the 'latest' checkpoint every N epochs instead of every "
            "epoch. Each save is a blocking torch.save() -- if --checkpoint_dir "
            "is on a networked filesystem, doing this every epoch can stall "
            "the GPU for a while with nothing queued for it. The final epoch "
            "always writes 'latest' regardless of this value, so it never "
            "goes stale relative to training. 'best' is unaffected -- it's "
            "still saved every time val loss improves."
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers. -1 = auto-detect from the job scheduler / CPU count.",
    )
    p.add_argument(
        "--no_shuffle_train",
        action="store_true",
        help="Disable shuffling of the training loader (notebook default was unshuffled).",
    )
    p.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Directory containing data_gen.py, zern.py, forward.py, rm.py, dual_branch_phasenet.py.",
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="TensorBoard run name. Defaults to a timestamp.",
    )

    args = p.parse_args()
    if args.init_from_zern is None:
        args.init_from_zern = args.zern_n
    return args


def resolve_num_workers(requested: int) -> int:
    if requested >= 0:
        return requested
    for var in ("PBS_NUM_PPN", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        val = os.environ.get(var)
        if val:
            try:
                return max(1, int(val))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    path, epoch, val_loss, best_val_loss, model, optimizer, scheduler, config
):
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


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):
    """Load a checkpoint into `model`, and optionally `optimizer`/`scheduler`.

    Pass optimizer=None and/or scheduler=None to skip restoring that piece of
    state (used for weights-only loading).
    """
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
# Scheduler factory
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, args):
    """Construct the LR scheduler named by --scheduler.

    Returns (scheduler, needs_metric), where needs_metric indicates whether
    scheduler.step() must be called with the val loss (ReduceLROnPlateau) or
    with no argument (everything else). To add a new schedule, add a branch
    here and a new --scheduler choice above.
    """
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.eta_min
        )
        return scheduler, False

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.eta_min,
        )
        return scheduler, True

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
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return scheduler, False

    raise ValueError(f"Unknown --scheduler: {args.scheduler!r}")


# --------------------------------------------------------------------------- #
# TensorBoard hparams logging
# --------------------------------------------------------------------------- #
def hparams_dict(args, run_name):
    """Flatten args into the {str: int|float|str|bool} shape TensorBoard's
    hparams plugin requires (no None, no Path, no list/dict)."""
    d = dict(vars(args))
    d.pop("run_name", None)
    d["run_name"] = run_name
    clean = {}
    for k, v in d.items():
        if v is None:
            continue
        clean[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return clean


# --------------------------------------------------------------------------- #
# Helpers ported from the notebook
# --------------------------------------------------------------------------- #
def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop the last two spatial dims of x down to (size, size)."""
    h, w = x.shape[-2], x.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top : top + size, left : left + size]


def compute_object_kernel(obj: torch.Tensor) -> torch.Tensor:
    """Ground-truth object kernel for PhaseRetrievalLoss.

    Uses the same k-space/real-space convention as the rest of the codebase:
        img_raw = ifftshift(ifft2(fftshift(image_k))) * 4.0
    so the forward direction (obj -> image_k) is:
        image_k = ifftshift(fft2(fftshift(obj))) / 4.0
    """
    obj_c = obj if torch.is_complex(obj) else obj.to(torch.complex64)
    shifted = torch.fft.fftshift(obj_c, dim=(-2, -1))
    Ok = torch.fft.fft2(shifted, dim=(-2, -1)) / 4.0
    Ok = torch.fft.ifftshift(Ok, dim=(-2, -1))
    Ok = Ok[..., 1:, 1:]
    return Ok


def align_global_phase(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rotate pred by the global phase that best aligns it with target."""
    inner_prod = torch.sum(pred * torch.conj(target), dim=(-2, -1), keepdim=True)
    theta = torch.angle(inner_prod)
    return pred * torch.exp(-1j * theta)


def phase_error_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    diff = torch.angle(pred * torch.conj(target)).abs()
    valid = mask.unsqueeze(0).expand_as(diff) > 0.5
    return diff[valid].mean().item(), diff[valid].max().item(), diff


def plot_comparison(pred, target, diff, mask, title, out_path, idx=0):
    p, t, m = pred[idx].cpu(), target[idx].cpu(), mask.cpu()
    fig, axes = plt.subplots(1, 3, figsize=(9, 8))
    axes[0].imshow(
        torch.where(m > 0.5, t.angle(), torch.nan),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[0].set_title(f"{title} target phase")
    axes[1].imshow(
        torch.where(m > 0.5, p.angle(), torch.nan),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1].set_title(f"{title} pred phase (global-phase aligned)")
    axes[2].imshow(
        torch.where(m > 0.5, diff[idx].cpu(), torch.nan),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[2].set_title(f"{title} angle difference")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
# A dedicated logger, rather than bare print(), for two reasons: (1) each
# handler below flushes on every emitted record, so nothing sits in a stdout
# buffer waiting to be written when the process is redirected to a file (as
# PBS does), and (2) it writes to its own file under log_dir/<run_name>/ --
# independent of PBS's train_out.log/train_err.log, which are typically only
# copied back to $PBS_O_WORKDIR once the job finishes (or not at all if it's
# killed). You can `tail -f <that file>` from a login node while the job runs.
logger = logging.getLogger("dual_branch_phasenet_train")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))  # still shows up in PBS's log


def setup_file_logging(log_path: Path):
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for h in logger.handlers:
        h.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    sys.path.insert(0, args.repo_root)
    from data_gen import RMDataset
    from zern import ZernikeAberration
    from forward import Simulation
    from rm import get_Rk_batched
    from dual_branch_phasenet import DualBranchPhaseNet, PhaseRetrievalLoss

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    log_dir = Path(args.log_dir).expanduser()
    for d in (checkpoint_dir, cache_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # run_name and file logging are set up first, before anything else can
    # fail, so a crash anywhere below still lands in a persistent log file.
    run_name = (
        args.run_name or f"DualBranchPhaseNet_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(run_dir / "train.log")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Logging to console and to {run_dir / 'train.log'}")

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    num_workers = resolve_num_workers(args.num_workers)
    logger.info(f"DataLoader workers: {num_workers}")

    N = args.N

    # ------------------------------------------------------------------- #
    # Datasets
    # ------------------------------------------------------------------- #
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # ------------------------------------------------------------------- #
    # Model / optimizer / scheduler
    # ------------------------------------------------------------------- #
    zern_gen = ZernikeAberration(N=N, zern_n=args.zern_n).to(device)
    simulation = Simulation(N, dtype=torch.complex64).to(device)

    with torch.no_grad():
        c_in_sample, _, _ = next(iter(train_dataloader))
        ab_sample = zern_gen(c_in_sample.to(device))
        ab_sample_cropped = center_crop(ab_sample, N)
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
    }

    start_epoch = 0
    best_val_loss = float("inf")

    # Checkpoints are kept per zern_n: increasing zern_n adds more aberration
    # modes to fit, which raises the achievable loss floor, so "best" from a
    # lower zern_n run is not a meaningful baseline for a higher one. This
    # run's own checkpoints (what gets written to) always live under
    # --zern_n. What gets *read* from at startup is governed separately by
    # --init_from_zern (defaults to --zern_n), so you can warm-start a run
    # at a new zern_n from a checkpoint trained at a different one.
    def ckpt_paths_for(zern_n):
        return {
            "latest": checkpoint_dir / f"dual_branch_phasenet_latest_zern{zern_n}.pth",
            "best": checkpoint_dir / f"dual_branch_phasenet_best_zern{zern_n}.pth",
        }

    ckpt_map = ckpt_paths_for(args.zern_n)

    if args.load_ckpt in ckpt_map:
        load_map = ckpt_paths_for(args.init_from_zern)
        ckpt_path = load_map[args.load_ckpt]
        cross_zern = args.init_from_zern != args.zern_n
        if ckpt_path.exists():
            if args.weights_only or cross_zern:
                # Only restore model weights; optimizer/scheduler start
                # fresh and epoch/best_val_loss are not carried over. This
                # is forced (regardless of --weights_only) whenever loading
                # across different zern_n, since epoch/optimizer/scheduler
                # state and best_val_loss from a different zern_n run don't
                # carry a meaningful interpretation here.
                load_checkpoint(
                    ckpt_path,
                    model,
                    optimizer=None,
                    scheduler=None,
                    map_location=device,
                )
                logger.info(
                    f"Loaded model weights only from '{ckpt_path.name}' "
                    f"(init_from_zern={args.init_from_zern}); starting at epoch 0 "
                    "with a fresh optimizer/scheduler and best_val_loss=inf."
                )
            else:
                start_epoch, _, best_val_loss, ckpt_config = load_checkpoint(
                    ckpt_path, model, optimizer, scheduler, map_location=device
                )
                saved_scheduler = ckpt_config.get("scheduler")
                if saved_scheduler is not None and saved_scheduler != args.scheduler:
                    logger.warning(
                        f"checkpoint was saved with --scheduler="
                        f"{saved_scheduler!r} but this run uses {args.scheduler!r}. "
                        "The scheduler state_dict was still loaded and may not "
                        "match the new scheduler's shape -- pass --weights_only "
                        "to skip that and start with a fresh scheduler instead."
                    )
                start_epoch += 1
                logger.info(
                    f"Resumed from '{ckpt_path.name}' (zern_n={args.zern_n}): "
                    f"epoch {start_epoch}, best_val_loss={best_val_loss:.4f}"
                )
        else:
            logger.info(
                f"--load_ckpt={args.load_ckpt} requested but {ckpt_path} "
                f"(init_from_zern={args.init_from_zern}) does not exist; "
                "training from scratch."
            )
    else:
        logger.info("Training from scratch.")

    criterion = PhaseRetrievalLoss(
        alpha_phase=args.alpha_phase, alpha_kernel=args.alpha_kernel
    )

    writer = SummaryWriter(str(log_dir / run_name))

    # Human-readable dump of every hyperparameter/CLI arg, written immediately
    # so it's captured even if the job is killed mid-run. Shows up under the
    # "Text" tab, scoped to this run.
    hparams = hparams_dict(args, run_name)
    writer.add_text(
        "config",
        "\n".join(f"- **{k}**: {v}" for k, v in sorted(hparams.items())),
    )

    # ------------------------------------------------------------------- #
    # Epoch loop
    # ------------------------------------------------------------------- #
    def run_epoch(dataloader, is_train, epoch):
        model.train() if is_train else model.eval()
        total_loss = 0.0

        with torch.set_grad_enabled(is_train):
            for c_in, c_out, obj in dataloader:
                c_in, c_out, obj = c_in.to(device), c_out.to(device), obj.to(device)

                with torch.no_grad():
                    ab_in = zern_gen(c_in)
                    ab_out = zern_gen(c_out)
                    k_outs = simulation(ab_in, ab_out, obj)
                    Rk = get_Rk_batched(
                        k_in=simulation.k_in_cropped, k_outs=k_outs, N=N
                    )
                    p_target = center_crop(ab_in, N)
                    q_target = center_crop(ab_out, N)
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
        phase = "train" if is_train else "val"
        writer.add_scalar(f"Loss/{phase}", avg_loss, epoch)
        return avg_loss

    logger.info("--- Starting Training ---")
    train_loss = val_loss = float("nan")  # in case start_epoch >= args.epochs
    for epoch in tqdm(range(start_epoch, args.epochs)):
        writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], epoch)
        train_loss = run_epoch(train_dataloader, is_train=True, epoch=epoch)
        val_loss = run_epoch(val_dataloader, is_train=False, epoch=epoch)
        scheduler.step(val_loss) if scheduler_needs_metric else scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        writer.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_map["best"],
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir
                / f"dual_branch_phasenet_epoch{epoch + 1}_zern{args.zern_n}.pth",
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

        is_last_epoch = epoch == args.epochs - 1
        if (epoch + 1) % args.save_every == 0 or is_last_epoch:
            save_checkpoint(
                ckpt_map["latest"],
                epoch,
                val_loss,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                config,
            )

    # Log final hparams alongside their end-of-run metrics, so different runs
    # can be compared side-by-side in the "HParams" tab. run_name="." keeps
    # this in the same event-file directory as the scalars above instead of
    # spawning a new nested run folder (add_hparams' default behavior).
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

    # ------------------------------------------------------------------- #
    # Final evaluation on the best checkpoint -> save plots to checkpoint_dir
    # ------------------------------------------------------------------- #
    load_checkpoint(
        ckpt_map["best"], model, optimizer=None, scheduler=None, map_location=device
    )
    model.eval()

    with torch.no_grad():
        c_in, c_out, obj = next(iter(val_dataloader))
        c_in, c_out, obj = c_in.to(device), c_out.to(device), obj.to(device)

        ab_in = zern_gen(c_in)
        ab_out = zern_gen(c_out)
        k_outs = simulation(ab_in, ab_out, obj)
        Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)
        p_target = center_crop(ab_in, N)
        q_target = center_crop(ab_out, N)

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
        checkpoint_dir / f"eval_p_comparison_zern{args.zern_n}.png",
    )
    plot_comparison(
        q_pred,
        q_target,
        q_diff,
        aperture_mask,
        "q",
        checkpoint_dir / f"eval_q_comparison_zern{args.zern_n}.png",
    )
    logger.info(f"Saved evaluation plots to {checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Guaranteed to reach the run's own log file (if setup_file_logging
        # ran) as well as stdout, regardless of what happens to PBS's own
        # train_out.log/train_err.log.
        logger.error(
            "Training crashed with an unhandled exception:\n%s", traceback.format_exc()
        )
        raise
