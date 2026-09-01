import argparse
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dual_branch_phasenet import DualBranchPhaseNet
from reconstruction_utils import get_dk_mapping, image_reconstruction
from zern import ZernikeAberration

N = 32
ZERN_N = 4
ZERN_NK = 15


def center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop spatial dimensions down to (size, size)."""
    h, w = x.shape[-2], x.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top : top + size, left : left + size]


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[DualBranchPhaseNet, torch.Tensor]:
    """Instantiate the model, load trained weights, and return aperture mask."""
    zern_gen = ZernikeAberration(N=N, zern_n=ZERN_N).to(device)
    dummy_c = torch.zeros((1, ZERN_NK), device=device)

    ab_sample = zern_gen(dummy_c)
    ab_sample_cropped = center_crop(ab_sample, N)
    aperture_mask = (ab_sample_cropped[0].abs() > 1e-6).float()

    model = DualBranchPhaseNet(N=N, embed_dim=256, aperture_mask=aperture_mask).to(
        device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, aperture_mask


def correct_phase(Rkk: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Apply phase conjugation to correct Rkk using predicted phase fields p and q."""
    B = Rkk.shape[0]
    q_conj = torch.conj(q).reshape(B, N * N, 1)  # k_out (rows)
    p_conj = torch.conj(p).reshape(B, 1, N * N)  # k_in (columns)

    return Rkk * q_conj * p_conj


def save_grid_summary(
    results: list[dict], save_path: Path, group_title: str, dpi: int = 300
) -> None:
    """
    Generates a single multi-row x 4-column high-resolution plot showing:
    [Phase p_pred | Phase q_pred | |img_raw| | |img_deconv|] for a grouped subset of inputs.
    """
    num_samples = len(results)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=4,
        figsize=(16.0, max(3.2 * num_samples, 3.2)),
        squeeze=False,
    )

    for row_idx, res in enumerate(results):
        filename = res["filename"]
        p_phase = (
            np.angle(res["p_pred"]) if np.iscomplexobj(res["p_pred"]) else res["p_pred"]
        )
        q_phase = (
            np.angle(res["q_pred"]) if np.iscomplexobj(res["q_pred"]) else res["q_pred"]
        )
        raw_mag = (
            np.abs(res["img_raw"])
            if np.iscomplexobj(res["img_raw"])
            else res["img_raw"]
        )
        deconv_mag = (
            np.abs(res["img_deconv"])
            if np.iscomplexobj(res["img_deconv"])
            else res["img_deconv"]
        )

        # --- Column 0: Predicted Phase p ---
        ax_p = axes[row_idx, 0]
        im0 = ax_p.imshow(
            p_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest"
        )
        ax_p.set_ylabel(
            filename, fontsize=9, fontweight="bold", rotation=0, ha="right", va="center"
        )
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        fig.colorbar(im0, ax=ax_p, fraction=0.046, pad=0.04)

        # --- Column 1: Predicted Phase q ---
        ax_q = axes[row_idx, 1]
        im1 = ax_q.imshow(
            q_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest"
        )
        ax_q.set_xticks([])
        ax_q.set_yticks([])
        fig.colorbar(im1, ax=ax_q, fraction=0.046, pad=0.04)

        # --- Column 2: Raw Image Magnitude ---
        ax_raw = axes[row_idx, 2]
        im2 = ax_raw.imshow(raw_mag, cmap="magma", interpolation="nearest")
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        fig.colorbar(im2, ax=ax_raw, fraction=0.046, pad=0.04)

        # --- Column 3: Deconvolved Image Magnitude ---
        ax_deconv = axes[row_idx, 3]
        im3 = ax_deconv.imshow(deconv_mag, cmap="magma", interpolation="nearest")
        ax_deconv.set_xticks([])
        ax_deconv.set_yticks([])
        fig.colorbar(im3, ax=ax_deconv, fraction=0.046, pad=0.04)

        # Column titles on top row
        if row_idx == 0:
            ax_p.set_title(
                "Predicted Phase p (rad)", fontsize=11, fontweight="bold", pad=10
            )
            ax_q.set_title(
                "Predicted Phase q (rad)", fontsize=11, fontweight="bold", pad=10
            )
            ax_raw.set_title("|img_raw|", fontsize=11, fontweight="bold", pad=10)
            ax_deconv.set_title("|img_deconv|", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(
        f"Reconstruction & Phase Retrieval Summary ({group_title})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DualBranchPhaseNet and generate layer-grouped visual summary plots"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/malmo/ml/data/data32",
        help="Directory with input .mat files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="~/malmo/ml/checkpoint/dual_branch_phasenet_best.pth",
        help="Path to .pth checkpoint file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="~/malmo/ml/data/results32",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--th", type=float, default=0.01, help="Threshold for OTF deconvolution"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI resolution for high-detail plot"
    )
    parser.add_argument(
        "--clean_phase",
        action="store_true",
        default=False,
        help="Set phase outside pupil aperture to 1.0 (zero phase) instead of model predictions",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    print(f"Clean phase outside aperture: {args.clean_phase}")

    mapping = get_dk_mapping(N=N, device=device)
    model, aperture_mask = load_model(checkpoint_path, device)

    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        return

    grouped_results = defaultdict(list)

    for path in mat_files:
        mat_data = sio.loadmat(path)
        Rkk_np = mat_data["Rkk"]

        Rkk_tensor = (
            torch.from_numpy(Rkk_np)
            .to(device=device, dtype=torch.complex64)
            .unsqueeze(0)
        )

        p_pred, q_pred, _ = model(Rkk_tensor)

        if args.clean_phase:
            # Mask out non-physical pixels outside aperture: set to 1.0 + 0.0j (phase = 0, unit mag)
            in_aperture = (aperture_mask > 1e-6).unsqueeze(0)
            unit_complex = torch.tensor(
                1.0 + 0.0j, device=device, dtype=torch.complex64
            )

            p_pred = torch.where(in_aperture, p_pred, unit_complex)
            q_pred = torch.where(in_aperture, q_pred, unit_complex)

        Rkk_corrected = correct_phase(Rkk_tensor, p_pred, q_pred)
        img_raw, img_deconv = image_reconstruction(
            Rkk_corrected, mapping, N=N, th=args.th
        )

        p_np = p_pred.squeeze(0).cpu().numpy()
        q_np = q_pred.squeeze(0).cpu().numpy()
        img_raw_np = img_raw.squeeze(0).cpu().numpy()
        img_deconv_np = img_deconv.squeeze(0).cpu().numpy()

        out_dict = {
            "p_pred": p_np,
            "q_pred": q_np,
            "Rkk_corrected": Rkk_corrected.squeeze(0).cpu().numpy(),
            "img_raw": img_raw_np,
            "img_deconv": img_deconv_np,
            "carrier": mat_data.get("carrier"),
            "wavelength_um": mat_data.get("wavelength_um"),
            "NA": mat_data.get("NA"),
        }
        out_mat_path = out_dir / f"eval_{path.name}"
        sio.savemat(out_mat_path, out_dict)

        match = re.search(r"(\d+layers?)", path.stem)
        group_key = match.group(1) if match else "other"

        grouped_results[group_key].append(
            {
                "filename": path.stem,
                "p_pred": p_np,
                "q_pred": q_np,
                "img_raw": img_raw_np,
                "img_deconv": img_deconv_np,
            }
        )

        print(f"Processed {path.name} -> Saved .mat to {out_mat_path.name}")

    for group_key, group_items in sorted(grouped_results.items()):
        summary_plot_path = out_dir / f"batch_summary_{group_key}.png"
        save_grid_summary(
            group_items, summary_plot_path, group_title=group_key, dpi=args.dpi
        )
        print(
            f"Saved {group_key} visual grid ({len(group_items)} samples) -> {summary_plot_path.name}"
        )


if __name__ == "__main__":
    main()
