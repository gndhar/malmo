"""
Evaluation script: run a trained model against a set of skimage sample
images (used as the scattering "object"), and for each one save:

  1. A phase-map comparison figure (true vs predicted input/output
     aberration).
  2. An image-reconstruction comparison figure (ML-corrected reconstruction
     vs ground-truth object).
  3. Covariance + normalized cross-correlation between the reconstructed
     image and the ground-truth object, appended to a summary CSV.

No training/run tracking here on purpose — this is a standalone eval pass
over a fixed model checkpoint.

--------------------------------------------------------------------------
ASSUMPTIONS TO DOUBLE-CHECK (marked inline with `# ASSUMPTION:`):
  - Model was saved with `torch.save(model.state_dict(), "model.pth")`,
    not the whole model object. Adjust the loading block if you pickled
    the whole `nn.Module` instead.
  - `zern_gen(coeffs)` returns a complex aberration tensor and, per your
    other script, may be oversampled (e.g. 2N x 2N) relative to N — the
    `get_clipped_phase` helper below crops to the central NxN region if
    needed. If your `ZernikeAberration` already returns NxN, this is a
    no-op crop.
  - `class_utils.get_dk_mapping` / `image_reconstruction` are the CLASS-
    algorithm utilities from your project (per earlier work) — swap in
    your actual import path if it differs.
--------------------------------------------------------------------------
"""

import os
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import data as skdata
from skimage.transform import resize
from skimage.color import rgb2gray

from zern import ZernikeAberration
from forward import Simulation
from rm import get_Rk_batched
from model_v2 import Model
from class_utils import (
    get_dk_mapping,
    image_reconstruction,
)  # ASSUMPTION: adjust if module differs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N = 16
ZERN_N = 5
MODEL_PATH = "model.pth"
OUTPUT_DIR = "eval_results"
SEED = 42  # fixes the aberration so every test image gets the same distortion

TEST_IMAGES = ["camera", "astronaut", "coins", "brick", "checkerboard", "chelsea"]

CMAP_PHASE = "twilight_shifted"
CMAP_IMAGE = "magma"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")

zern_gen = ZernikeAberration(N=N, zern_n=ZERN_N)
coeff_count = zern_gen.num_coefficients
simulation = Simulation(N, dtype=torch.complex64)

model = Model(N, coeff_count).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Device: {device} | N: {N} | coeff_count: {coeff_count}")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_object(name: str, size: int) -> torch.Tensor:
    """Load a skimage sample image, convert to grayscale, resize to
    (size, size), normalize to [0, 1], and cast to complex64 (object
    field with zero imaginary part)."""
    img = getattr(skdata, name)()
    if img.ndim == 3:
        img = rgb2gray(img)
    img = resize(img, (size, size), anti_aliasing=True)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    obj = torch.tensor(img, dtype=torch.complex64)
    return obj


def get_clipped_phase(ab: torch.Tensor, N: int) -> np.ndarray:
    """Extract central N x N phase from a possibly-oversampled aberration
    map. No-op crop if `ab` is already N x N."""
    phase = np.angle(ab.detach().cpu().numpy())
    h, w = phase.shape[-2], phase.shape[-1]
    start_h, start_w = (h - N) // 2, (w - N) // 2
    if start_h <= 0 and start_w <= 0:
        return phase
    return phase[..., start_h : start_h + N, start_w : start_w + N]


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two images (order-of-
    magnitude-invariant version of covariance)."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def covariance(a: np.ndarray, b: np.ndarray) -> float:
    """Raw covariance between two images. Scale-dependent (unlike the
    normalized correlation above) — useful mainly for comparing runs at
    a fixed intensity normalization."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.mean((a - a.mean()) * (b - b.mean())))


def match_shapes(a: np.ndarray, b: np.ndarray) -> tuple:
    """Resize the larger array down to the smaller one's shape so metrics
    can be computed even if reconstruction and object grids differ."""
    if a.shape == b.shape:
        return a, b
    target = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    a_r = resize(a, target, anti_aliasing=True)
    b_r = resize(b, target, anti_aliasing=True)
    return a_r, b_r


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------
def evaluate_one(image_name: str, summary_rows: list):
    torch.manual_seed(SEED)  # same aberration draw for every test image

    obj_size = 2 * N  # ASSUMPTION: matches the 2N object grid used in training
    obj = load_test_object(image_name, obj_size).to(device)

    c_in = torch.rand(1, coeff_count, device=device)
    c_out = torch.rand(1, coeff_count, device=device)

    with torch.no_grad():
        ab_in = zern_gen(c_in)
        ab_out = zern_gen(c_out)
        k_outs = simulation(ab_in, ab_out, obj.unsqueeze(0))
        Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)

        pred_in, pred_out = model(Rk)

    # --- phase maps ---
    true_in_phase = get_clipped_phase(ab_in[0], N)
    true_out_phase = get_clipped_phase(ab_out[0], N)
    pred_ab_in = zern_gen(pred_in)
    pred_ab_out = zern_gen(pred_out)
    pred_in_phase = get_clipped_phase(pred_ab_in[0], N)
    pred_out_phase = get_clipped_phase(pred_ab_out[0], N)

    img_dir = os.path.join(OUTPUT_DIR, image_name)
    os.makedirs(img_dir, exist_ok=True)

    fig1, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig1.suptitle(f"True vs Predicted Phase Maps — {image_name}", fontsize=16)
    for ax, phase, title in zip(
        axs.flat,
        [true_in_phase, pred_in_phase, true_out_phase, pred_out_phase],
        [
            "True Input Aberration",
            "Predicted Input Aberration",
            "True Output Aberration",
            "Predicted Output Aberration",
        ],
    ):
        im = ax.imshow(phase, cmap=CMAP_PHASE, vmin=-np.pi, vmax=np.pi)
        ax.set_title(title)
        ax.axis("off")
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    plt.tight_layout()
    fig1.savefig(os.path.join(img_dir, "phase_comparison.png"), dpi=150)
    plt.close(fig1)

    # --- corrected reconstruction ---
    Rk_np = Rk[0].detach().cpu().numpy().reshape(N * N, N * N)
    ab_in_corr = np.exp(-1j * pred_in_phase).flatten()
    ab_out_corr = np.exp(-1j * pred_out_phase).flatten()
    Rk_corrected = (Rk_np * ab_in_corr[None, :]) * ab_out_corr[:, None]

    mapping = get_dk_mapping(N)
    _, ml_image = image_reconstruction(Rk_corrected, mapping, N)
    _, uncorrected_image = image_reconstruction(Rk_np, mapping, N)

    ml_mag = np.abs(ml_image)
    obj_mag = np.abs(obj.detach().cpu().numpy())
    ml_mag_m, obj_mag_m = match_shapes(ml_mag, obj_mag)

    corr = normalized_cross_correlation(ml_mag_m, obj_mag_m)
    cov = covariance(ml_mag_m, obj_mag_m)
    coeff_mse = float(
        np.mean(
            (
                torch.cat([pred_in, pred_out], dim=1).detach().cpu().numpy()
                - torch.cat([c_in, c_out], dim=1).detach().cpu().numpy()
            )
            ** 2
        )
    )

    fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(f"Image Reconstruction — {image_name}", fontsize=14)
    axs2[0].imshow(obj_mag, cmap=CMAP_IMAGE)
    axs2[0].set_title("Ground Truth Object")
    axs2[0].axis("off")
    axs2[1].imshow(np.abs(uncorrected_image), cmap=CMAP_IMAGE)
    axs2[1].set_title("Uncorrected Reconstruction")
    axs2[1].axis("off")
    axs2[2].imshow(ml_mag, cmap=CMAP_IMAGE)
    axs2[2].set_title(f"ML-Corrected (corr={corr:.3f})")
    axs2[2].axis("off")
    plt.tight_layout()
    fig2.savefig(os.path.join(img_dir, "image_comparison.png"), dpi=150)
    plt.close(fig2)

    print(f"[{image_name}] coeff_mse={coeff_mse:.6f}  corr={corr:.4f}  cov={cov:.6f}")
    summary_rows.append(
        {
            "image": image_name,
            "coeff_mse": coeff_mse,
            "correlation": corr,
            "covariance": cov,
        }
    )


# ---------------------------------------------------------------------------
# Run over all test images + write summary
# ---------------------------------------------------------------------------
summary_rows = []
for name in TEST_IMAGES:
    # try:
    evaluate_one(name, summary_rows)
    # except Exception as e:
    #     print(f"[{name}] FAILED: {e}")

summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "coeff_mse", "correlation", "covariance"]
    )
    writer.writeheader()
    writer.writerows(summary_rows)
print(f"\nSaved summary: {summary_path}")

if summary_rows:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    names = [r["image"] for r in summary_rows]
    corrs = [r["correlation"] for r in summary_rows]
    ax3.bar(names, corrs, color="steelblue")
    ax3.set_ylabel("Correlation (reconstruction vs. object)")
    ax3.set_title("Reconstruction quality across test images")
    ax3.set_ylim(-1, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "summary_correlation.png"), dpi=150)
    plt.close(fig3)
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'summary_correlation.png')}")
