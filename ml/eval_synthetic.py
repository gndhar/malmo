"""Evaluation Script for DualBranchPhaseNet on Synthetic Validation Data.

Features:
1. Phase map recovery evaluation (p, q targets vs predictions with global phase alignment).
2. Complex object reconstruction (Ground Truth, Ideal Phase-Corrected, Raw Pred, Deconvolved Pred).
3. Robust SNR Sensitivity Analysis sweeping from +20 dB down to -20 dB (including Ideal upper bound).
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_gen import RMDataset
from dual_branch_phasenet import DualBranchPhaseNet
from forward import Simulation
from reconstruction_utils import get_dk_mapping, image_reconstruction
from rm import get_Rk_batched


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crops spatial dimensions down to (size, size)."""
    h, w = x.shape[-2], x.shape[-1]
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top : top + size, left : left + size]


def align_global_phase(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Aligns global phase offset between complex predictions and ground truth targets."""
    inner_prod = torch.sum(pred * torch.conj(target), dim=(-2, -1), keepdim=True)
    return pred * torch.exp(-1j * torch.angle(inner_prod))


def correct_phase(Rkk: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Applies phase conjugation to correct Rkk matrix using phase fields p and q."""
    B, N2, _ = Rkk.shape
    q_conj = torch.conj(q).reshape(B, N2, 1)  # k_out (rows)
    p_conj = torch.conj(p).reshape(B, 1, N2)  # k_in (columns)
    return Rkk * q_conj * p_conj


def add_complex_noise(Rk: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Adds complex Additive White Gaussian Noise (AWGN) to match target SNR (dB)."""
    if math.isinf(snr_db) or snr_db is None:
        return Rk
    sig_power = torch.mean(torch.abs(Rk) ** 2, dim=(-2, -1), keepdim=True)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    noise_std = torch.sqrt(noise_power / 2.0)
    noise_real = torch.randn_like(Rk.real) * noise_std
    noise_imag = torch.randn_like(Rk.imag) * noise_std
    return Rk + torch.complex(noise_real, noise_imag)


def compute_phase_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    """Computes Mean Absolute Phase Error (in radians) inside pupil aperture."""
    pred_aligned = align_global_phase(pred, target)
    diff = torch.angle(pred_aligned * torch.conj(target)).abs()
    valid_mask = mask.unsqueeze(0).expand_as(diff) > 0.5
    return diff[valid_mask].mean().item()


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) for magnitude images normalized to [0, 1]."""
    p_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    t_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
    mse = np.mean((p_norm - t_norm) ** 2)
    if mse < 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


# --------------------------------------------------------------------------- #
# Visualization Helpers
# --------------------------------------------------------------------------- #
def plot_phase_maps(
    p_target: torch.Tensor,
    p_pred: torch.Tensor,
    q_target: torch.Tensor,
    q_pred: torch.Tensor,
    aperture_mask: torch.Tensor,
    out_path: Path,
    idx: int = 0,
) -> None:
    """Plots comparison of target vs predicted phase maps for p and q."""
    p_t = p_target[idx].cpu()
    p_p = align_global_phase(p_pred[idx : idx + 1], p_target[idx : idx + 1])[0].cpu()
    q_t = q_target[idx].cpu()
    q_p = align_global_phase(q_pred[idx : idx + 1], q_target[idx : idx + 1])[0].cpu()
    mask = aperture_mask.cpu() > 0.5

    p_err = torch.angle(p_p * torch.conj(p_t)).abs()
    q_err = torch.angle(q_p * torch.conj(q_t)).abs()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    panels = [
        (axes[0, 0], p_t.angle(), r"$p$ Target Phase", "twilight", -np.pi, np.pi),
        (axes[0, 1], p_p.angle(), r"$p$ Pred Phase", "twilight", -np.pi, np.pi),
        (axes[0, 2], p_err, r"$p$ Absolute Error", "magma", 0, np.pi),
        (axes[1, 0], q_t.angle(), r"$q$ Target Phase", "twilight", -np.pi, np.pi),
        (axes[1, 1], q_p.angle(), r"$q$ Pred Phase", "twilight", -np.pi, np.pi),
        (axes[1, 2], q_err, r"$q$ Absolute Error", "magma", 0, np.pi),
    ]

    for ax, data, title, cmap, vmin, vmax in panels:
        masked_data = torch.where(mask, data, torch.nan).numpy()
        im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Validation Phase Map Recovery", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(
    obj_gt: torch.Tensor,
    img_ideal: torch.Tensor,
    img_raw: torch.Tensor,
    img_deconv: torch.Tensor,
    out_path: Path,
    idx: int = 0,
) -> None:
    """Plots GT target, Ideal phase-corrected object, Raw Pred, & Deconvolved Pred objects."""
    gt_mag = obj_gt[idx].abs().cpu().numpy()
    ideal_mag = img_ideal[idx].abs().cpu().numpy()
    raw_mag = img_raw[idx].abs().cpu().numpy()
    deconv_mag = img_deconv[idx].abs().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (axes[0], gt_mag, "Ground Truth Object"),
        (
            axes[1],
            ideal_mag,
            f"Ideal Object (GT Phase)\n(PSNR: {compute_psnr(ideal_mag, gt_mag):.2f} dB)",
        ),
        (
            axes[2],
            raw_mag,
            f"Pred Raw Reconstruction\n(PSNR: {compute_psnr(raw_mag, gt_mag):.2f} dB)",
        ),
        (
            axes[3],
            deconv_mag,
            f"Pred Deconvolved Object\n(PSNR: {compute_psnr(deconv_mag, gt_mag):.2f} dB)",
        ),
    ]

    for ax, data, title in panels:
        im = ax.imshow(data, cmap="magma")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Validation Object Reconstruction Summary", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_snr_sweep(
    snr_levels: List[float],
    phase_p_maes: List[float],
    phase_q_maes: List[float],
    psnr_ideals: List[float],
    psnr_raws: List[float],
    psnr_deconvs: List[float],
    out_path: Path,
) -> None:
    """Plots SNR sensitivity analysis metrics for phase errors and object reconstruction PSNRs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Phase MAE Plot
    ax1.plot(
        snr_levels,
        phase_p_maes,
        "o-",
        label="Phase p MAE",
        color="crimson",
        linewidth=2,
    )
    ax1.plot(
        snr_levels,
        phase_q_maes,
        "s--",
        label="Phase q MAE",
        color="darkorange",
        linewidth=2,
    )
    ax1.set_xlabel("SNR (dB)", fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (rad)", fontweight="bold")
    ax1.set_title("Phase Retrieval Accuracy vs SNR", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    ax1.invert_xaxis()  # 20dB on left to -20dB on right

    # Object PSNR Plot
    ax2.plot(
        snr_levels,
        psnr_ideals,
        "^-.",
        label="Ideal Object PSNR (GT Phase)",
        color="forestgreen",
        linewidth=2,
    )
    ax2.plot(
        snr_levels,
        psnr_deconvs,
        "o-",
        label="Pred Deconvolved PSNR",
        color="teal",
        linewidth=2,
    )
    ax2.plot(
        snr_levels,
        psnr_raws,
        "s--",
        label="Pred Raw PSNR",
        color="steelblue",
        linewidth=2,
    )
    ax2.set_xlabel("SNR (dB)", fontweight="bold")
    ax2.set_ylabel("PSNR (dB)", fontweight="bold")
    ax2.set_title("Object Reconstruction Quality vs SNR", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    ax2.invert_xaxis()

    fig.suptitle(
        "Performance Breakdown across Noise Levels (SNR Sweep)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main Validation Pipeline
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DualBranchPhaseNet on Validation Data"
    )
    parser.add_argument(
        "--N", type=int, default=32, help="Grid spatial resolution N x N"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model .pth checkpoint",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/validation_eval",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--val_size", type=int, default=64, help="Validation set evaluation size"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluation batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=420, help="Random seed for validation dataset"
    )
    parser.add_argument(
        "--th", type=float, default=0.01, help="Threshold for OTF deconvolution"
    )
    parser.add_argument(
        "--snr_min",
        type=float,
        default=-20.0,
        help="Minimum SNR for sweep analysis (dB)",
    )
    parser.add_argument(
        "--snr_max",
        type=float,
        default=20.0,
        help="Maximum SNR for sweep analysis (dB)",
    )
    parser.add_argument(
        "--snr_steps", type=int, default=9, help="Number of SNR evaluation steps"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Validation Evaluation on device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    N = args.N

    # 1. Create Validation Dataset & Loader FIRST — the aperture mask (below) is
    #    derived from an actual sample, so the dataset must exist before it.
    val_dataset = RMDataset(
        N=2 * N,
        size=args.val_size,
        seed=args.seed,
        cache_objects=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Derive aperture mask directly from dataset output phase map (mirrors
    #    train.py exactly: center-crop a real ab_in sample down to N and
    #    threshold it, rather than building an independent ZernikeAberration
    #    pupil at the wrong resolution). This is what keeps the mask's shape
    #    and support consistent with p_target/q_target and with what the
    #    model was actually trained against.
    with torch.no_grad():
        ab_in_sample, _, _ = next(iter(val_loader))
        ab_sample_cropped = center_crop(ab_in_sample.to(device), N)
        aperture_mask = (ab_sample_cropped[0].abs() > 1e-6).float()

    # 3. Load Model
    model = DualBranchPhaseNet(N=N, embed_dim=256, aperture_mask=aperture_mask).to(
        device
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Physics Simulation & Reconstruction Mapping
    simulation = Simulation(N, dtype=torch.complex64).to(device)
    mapping = get_dk_mapping(N=N, device=device)

    # ----------------------------------------------------------------------- #
    # Section 1 & 2: Clean Performance Evaluation & Visualization
    # ----------------------------------------------------------------------- #
    print("\n--- Evaluating Phase Map & Reconstruction Quality (Clean Data) ---")
    all_p_target, all_p_pred = [], []
    all_q_target, all_q_pred = [], []
    all_obj_gt, all_img_ideal, all_img_raw, all_img_deconv = [], [], [], []

    with torch.no_grad():
        for ab_in, ab_out, obj in tqdm(val_loader, desc="Clean Eval"):
            ab_in, ab_out, obj = ab_in.to(device), ab_out.to(device), obj.to(device)

            k_outs = simulation(ab_in, ab_out, obj)
            Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)

            p_target = center_crop(ab_in, N)
            q_target = center_crop(ab_out, N)
            # NOTE: obj (and the reconstructed images below) live on the
            # doubled k-space grid (-2*NA to 2*NA), matching the object's
            # native resolution — only the predicted phase p/q is cropped
            # to the model's -NA..NA range. Do NOT crop obj_gt to N here.
            obj_gt = obj

            # Ideal Reconstruction (using GT Phase)
            Rkk_ideal = correct_phase(Rk, p_target, q_target)
            _, img_ideal = image_reconstruction(Rkk_ideal, mapping, N=N, th=args.th)

            # Predicted Reconstruction (using Predicted Phase)
            p_pred, q_pred, _ = model(Rk)
            Rkk_corrected = correct_phase(Rk, p_pred, q_pred)
            img_raw, img_deconv = image_reconstruction(
                Rkk_corrected, mapping, N=N, th=args.th
            )

            all_p_target.append(p_target)
            all_p_pred.append(p_pred)
            all_q_target.append(q_target)
            all_q_pred.append(q_pred)
            all_obj_gt.append(obj_gt)
            all_img_ideal.append(img_ideal)
            all_img_raw.append(img_raw)
            all_img_deconv.append(img_deconv)

    p_targets = torch.cat(all_p_target, dim=0)
    p_preds = torch.cat(all_p_pred, dim=0)
    q_targets = torch.cat(all_q_target, dim=0)
    q_preds = torch.cat(all_q_pred, dim=0)
    objs_gt = torch.cat(all_obj_gt, dim=0)
    imgs_ideal = torch.cat(all_img_ideal, dim=0)
    imgs_raw = torch.cat(all_img_raw, dim=0)
    imgs_deconv = torch.cat(all_img_deconv, dim=0)

    # Compute Overall Clean Metrics
    p_mae = compute_phase_mae(p_preds, p_targets, aperture_mask)
    q_mae = compute_phase_mae(q_preds, q_targets, aperture_mask)

    ideal_psnrs = [
        compute_psnr(i.abs().cpu().numpy(), g.abs().cpu().numpy())
        for i, g in zip(imgs_ideal, objs_gt)
    ]
    raw_psnrs = [
        compute_psnr(r.abs().cpu().numpy(), g.abs().cpu().numpy())
        for r, g in zip(imgs_raw, objs_gt)
    ]
    deconv_psnrs = [
        compute_psnr(d.abs().cpu().numpy(), g.abs().cpu().numpy())
        for d, g in zip(imgs_deconv, objs_gt)
    ]

    print(f"Clean Phase p MAE       : {p_mae:.4f} rad")
    print(f"Clean Phase q MAE       : {q_mae:.4f} rad")
    print(f"Clean Ideal Image PSNR  : {np.mean(ideal_psnrs):.2f} dB")
    print(f"Clean Pred Raw PSNR     : {np.mean(raw_psnrs):.2f} dB")
    print(f"Clean Pred Deconv PSNR  : {np.mean(deconv_psnrs):.2f} dB")

    # Generate Clean Visualizations
    plot_phase_maps(
        p_targets,
        p_preds,
        q_targets,
        q_preds,
        aperture_mask,
        out_dir / "val_phase_comparison.png",
    )
    plot_reconstructions(
        objs_gt,
        imgs_ideal,
        imgs_raw,
        imgs_deconv,
        out_dir / "val_object_reconstruction.png",
    )
    print(f"Saved phase map & reconstruction plots to {out_dir}")

    # ----------------------------------------------------------------------- #
    # Section 3: SNR Sensitivity Analysis (Sweep 20 dB to -20 dB)
    # ----------------------------------------------------------------------- #
    print(
        f"\n--- Running SNR Analysis ({args.snr_max} dB down to {args.snr_min} dB) ---"
    )
    snr_levels = np.linspace(args.snr_max, args.snr_min, args.snr_steps).tolist()

    sweep_p_maes, sweep_q_maes = [], []
    sweep_ideal_psnrs, sweep_raw_psnrs, sweep_deconv_psnrs = [], [], []

    for snr in snr_levels:
        batch_p_maes, batch_q_maes = [], []
        batch_ideal_psnrs, batch_raw_psnrs, batch_deconv_psnrs = [], [], []

        with torch.no_grad():
            for ab_in, ab_out, obj in val_loader:
                ab_in, ab_out, obj = ab_in.to(device), ab_out.to(device), obj.to(device)

                k_outs = simulation(ab_in, ab_out, obj)
                Rk_clean = get_Rk_batched(
                    k_in=simulation.k_in_cropped, k_outs=k_outs, N=N
                )

                # Add noise matching target SNR
                Rk_noisy = add_complex_noise(Rk_clean, snr_db=snr)

                p_target = center_crop(ab_in, N)
                q_target = center_crop(ab_out, N)
                obj_gt = obj  # full-resolution, matches image_reconstruction output

                # Ideal phase correction on noisy Rk (theoretical upper bound)
                Rkk_ideal = correct_phase(Rk_noisy, p_target, q_target)
                _, img_ideal = image_reconstruction(Rkk_ideal, mapping, N=N, th=args.th)

                # Model prediction on noisy Rk
                p_pred, q_pred, _ = model(Rk_noisy)

                # Predicted phase correction & reconstruction
                Rkk_corrected = correct_phase(Rk_noisy, p_pred, q_pred)
                img_raw, img_deconv = image_reconstruction(
                    Rkk_corrected, mapping, N=N, th=args.th
                )

                # Metrics computation
                batch_p_maes.append(compute_phase_mae(p_pred, p_target, aperture_mask))
                batch_q_maes.append(compute_phase_mae(q_pred, q_target, aperture_mask))

                for i, r, d, g in zip(img_ideal, img_raw, img_deconv, obj_gt):
                    g_np = g.abs().cpu().numpy()
                    batch_ideal_psnrs.append(compute_psnr(i.abs().cpu().numpy(), g_np))
                    batch_raw_psnrs.append(compute_psnr(r.abs().cpu().numpy(), g_np))
                    batch_deconv_psnrs.append(compute_psnr(d.abs().cpu().numpy(), g_np))

        avg_p_mae = np.mean(batch_p_maes)
        avg_q_mae = np.mean(batch_q_maes)
        avg_ideal_psnr = np.mean(batch_ideal_psnrs)
        avg_raw_psnr = np.mean(batch_raw_psnrs)
        avg_deconv_psnr = np.mean(batch_deconv_psnrs)

        sweep_p_maes.append(avg_p_mae)
        sweep_q_maes.append(avg_q_mae)
        sweep_ideal_psnrs.append(avg_ideal_psnr)
        sweep_raw_psnrs.append(avg_raw_psnr)
        sweep_deconv_psnrs.append(avg_deconv_psnr)

        print(
            f"SNR: {snr:5.1f} dB | "
            f"Phase p MAE: {avg_p_mae:.4f} rad | "
            f"Phase q MAE: {avg_q_mae:.4f} rad | "
            f"Ideal PSNR: {avg_ideal_psnr:.2f} dB | "
            f"Deconv PSNR: {avg_deconv_psnr:.2f} dB"
        )

    # Generate SNR Sweep Plots
    plot_snr_sweep(
        snr_levels,
        sweep_p_maes,
        sweep_q_maes,
        sweep_ideal_psnrs,
        sweep_raw_psnrs,
        sweep_deconv_psnrs,
        out_dir / "snr_sensitivity_analysis.png",
    )
    print(f"Saved SNR sensitivity plot to {out_dir / 'snr_sensitivity_analysis.png'}")


if __name__ == "__main__":
    main()
