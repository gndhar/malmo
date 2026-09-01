import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root malmo package directory is in sys.path
current_dir = Path(__file__).resolve().parent
malmo_root = current_dir.parent if current_dir.name == "ml" else current_dir
if str(malmo_root) not in sys.path:
    sys.path.insert(0, str(malmo_root))

from class_utils import class_algorithm

N = 32


def save_grid_summary(
    results: list[dict], save_path: Path, group_title: str, dpi: int = 300
) -> None:
    """
    Generates a single multi-row x 3-column plot showing CLASS outputs per sample:
    [ Estimated θ_in (rad) | Estimated θ_out (rad) | Reconstructed Image (CLASS Deconv) ]
    """
    num_samples = len(results)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=3,
        figsize=(12.0, max(3.5 * num_samples, 3.5)),
        squeeze=False,
    )

    for row_idx, res in enumerate(results):
        filename = res["filename"]
        ab_in_phase = -np.angle(res["ab_in"])
        ab_out_phase = -np.angle(res["ab_out"])
        img_deconv_mag = np.abs(res["final_image"])

        # Column 0: Input Phase Aberration (theta_in)
        ax_in = axes[row_idx, 0]
        im0 = ax_in.imshow(
            ab_in_phase,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
        )
        ax_in.set_ylabel(
            filename, fontsize=9, fontweight="bold", rotation=0, ha="right", va="center"
        )
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        fig.colorbar(im0, ax=ax_in, fraction=0.046, pad=0.04)

        # Column 1: Output Phase Aberration (theta_out)
        ax_out = axes[row_idx, 1]
        im1 = ax_out.imshow(
            ab_out_phase,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
        )
        ax_out.set_xticks([])
        ax_out.set_yticks([])
        fig.colorbar(im1, ax=ax_out, fraction=0.046, pad=0.04)

        # Column 2: CLASS Deconvolved Reconstructed Image
        ax_img = axes[row_idx, 2]
        im2 = ax_img.imshow(img_deconv_mag, cmap="magma", interpolation="nearest")
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        fig.colorbar(im2, ax=ax_img, fraction=0.046, pad=0.04)

        # Top headers
        if row_idx == 0:
            ax_in.set_title(
                "Input aberration (θ_in)", fontsize=11, fontweight="bold", pad=10
            )
            ax_out.set_title(
                "Output aberration (θ_out)", fontsize=11, fontweight="bold", pad=10
            )
            ax_img.set_title(
                "Reconstructed image (CLASS Deconv)",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )

    fig.suptitle(
        f"CLASS Baseline Reconstructions ({group_title})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run classical CLASS algorithm baseline on mat files"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/malmo/ml/data/data32",
        help="Input .mat dataset directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="~/malmo/ml/data/results32_class",
        help="Output directory to save mat and plot results",
    )
    parser.add_argument(
        "--max_iter", type=int, default=10, help="Max CLASS iteration steps"
    )
    parser.add_argument(
        "--max_pi", type=int, default=6, help="Max Power Iteration steps"
    )
    parser.add_argument(
        "--kfilter", type=int, default=6, help="Kernel filter cutoff radius"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI resolution for saved figures"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(data_dir.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        return

    grouped_results = defaultdict(list)

    for path in mat_files:
        mat_data = sio.loadmat(path)
        Rkk = mat_data["Rkk"]

        # Run classical CLASS algorithm (returns deconvolved final_image, ab_in, ab_out)
        final_image, ab_in, ab_out = class_algorithm(
            Rkk,
            N=N,
            max_iteration_number=args.max_iter,
            max_PI_num=args.max_pi,
            kfilter=args.kfilter,
        )

        out_dict = {
            "img_deconv": final_image,
            "ab_in": ab_in,
            "ab_out": ab_out,
            "carrier": mat_data.get("carrier"),
            "wavelength_um": mat_data.get("wavelength_um"),
            "NA": mat_data.get("NA"),
        }
        out_mat_path = out_dir / f"eval_class_{path.name}"
        sio.savemat(out_mat_path, out_dict)

        match = re.search(r"(\d+layers?)", path.stem)
        group_key = match.group(1) if match else "all"

        grouped_results[group_key].append(
            {
                "filename": path.stem,
                "final_image": final_image,
                "ab_in": ab_in,
                "ab_out": ab_out,
            }
        )

        print(f"Processed {path.name} -> Saved outputs to {out_mat_path.name}")

    for group_key, group_items in sorted(grouped_results.items()):
        summary_plot_path = out_dir / f"class_summary_{group_key}.png"
        save_grid_summary(
            group_items, summary_plot_path, group_title=group_key, dpi=args.dpi
        )
        print(f"Saved {group_key} visual grid -> {summary_plot_path.name}")


if __name__ == "__main__":
    main()
