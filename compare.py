import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Must match training exactly ---
from config import config

config.N = 16  # FIX 1: set explicitly, just like training does

import zern
from resnet import ResNetEstimator
from class_utils import get_dk_mapping, image_reconstruction
from forward_pt import simulate_pt_vectorized
from reflection_pt import generate_R_k_pt
import obj

# ---------------------------------------------------------------------------
# 1. Setup  (mirrors training script exactly)
# ---------------------------------------------------------------------------
N = config.N
ratio = 0.5
coeff_count = int(zern.cart.nk * ratio)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device} | N: {N} | coeff_count: {coeff_count}")

# ---------------------------------------------------------------------------
# 2. Load model
#    FIX 2: training used torch.save(model, "model.pth")  — whole model, not
#    state_dict — so we load it the same way. Also the filename is model.pth,
#    not model_weights.pth.
# ---------------------------------------------------------------------------
model = torch.load("model.pth", map_location=device, weights_only=False)
model.eval()
print("Model loaded.")


# ---------------------------------------------------------------------------
# 3. generate_training_batch — copied verbatim from the Colab notebook
#    FIX 3: this function is NOT in any importable module; it was defined
#    inline in the notebook, so we reproduce it here exactly.
# ---------------------------------------------------------------------------
def generate_training_batch(batch_size, coeff_count, device):
    # 1. Generate coefficients on CPU using NumPy so zern.py can read them
    c_in_np = np.random.rand(batch_size, coeff_count).astype(np.float32)
    c_out_np = np.random.rand(batch_size, coeff_count).astype(np.float32)

    # 2. Vectorized GPU Simulation (Now handles batches)
    k_ins, k_outs = simulate_pt_vectorized(c_in_np, c_out_np, device)

    # 3. Generate Reflection Matrices natively on GPU
    R_k = generate_R_k_pt(k_ins, k_outs)

    # 4. Format for ResNet input (batch_size, N, N, N, N)
    inputs = R_k.reshape(batch_size, N, N, N, N)

    # Target tensors move directly to device
    targets = torch.cat(
        (torch.tensor(c_in_np, device=device), torch.tensor(c_out_np, device=device)),
        dim=1,
    )
    return inputs, targets


# ---------------------------------------------------------------------------
# 4. Run evaluation — identical to the Colab evaluate_and_visualize cell
# ---------------------------------------------------------------------------
with torch.no_grad():
    inputs, targets = generate_training_batch(1, coeff_count, device)
    preds = model(inputs)

c_true = targets[0].cpu().numpy()
c_pred = preds[0].cpu().numpy()

c_in_true = c_true[:coeff_count]
c_out_true = c_true[coeff_count:]
c_in_pred = c_pred[:coeff_count]
c_out_pred = c_pred[coeff_count:]

print("\nInput Coefficients (c_in):")
print(f"  True: {np.round(c_in_true, 4)}")
print(f"  Pred: {np.round(c_in_pred, 4)}")
print("\nOutput Coefficients (c_out):")
print(f"  True: {np.round(c_out_true, 4)}")
print(f"  Pred: {np.round(c_out_pred, 4)}")
print(f"\nMSE: {np.mean((c_pred - c_true) ** 2):.6f}")

# ---------------------------------------------------------------------------
# 5. Phase maps
# ---------------------------------------------------------------------------
from zern import generate_abberations

in_abb_true = np.angle(generate_abberations(c_in_true))
out_abb_true = np.angle(generate_abberations(c_out_true))
in_abb_pred = np.angle(generate_abberations(c_in_pred))
out_abb_pred = np.angle(generate_abberations(c_out_pred))


# ---------------------------------------------------------------------------
# 6. Apply ML correction and reconstruct image
# ---------------------------------------------------------------------------
def get_clipped_phase(abb_2N, N):
    """Extract central N x N phase from a 2N x 2N aberration map."""
    phase_2N = np.angle(abb_2N)
    start = N // 2
    return phase_2N[start : start + N, start : start + N]


est_phase_in = get_clipped_phase(generate_abberations(c_in_pred), N)
est_phase_out = get_clipped_phase(generate_abberations(c_out_pred), N)

Rk_np = inputs[0].cpu().numpy().reshape(N * N, N * N)
ab_in_cor = np.exp(-1j * est_phase_in).flatten()
ab_out_cor = np.exp(-1j * est_phase_out).flatten()
Rk_corrected = (Rk_np * ab_in_cor[None, :]) * ab_out_cor[:, None]

mapping = get_dk_mapping(N)
_, ml_image = image_reconstruction(Rk_corrected, mapping, N)

# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
os.makedirs("ml_results", exist_ok=True)
cmap = "twilight_shifted"

# Phase map comparison
fig1, axs = plt.subplots(2, 2, figsize=(10, 10))
fig1.suptitle("True vs Predicted Zernike Phase Maps", fontsize=16)

for ax, data, title in zip(
    axs.flat,
    [in_abb_true, in_abb_pred, out_abb_true, out_abb_pred],
    [
        "True Input Aberration",
        "Predicted Input Aberration",
        "True Output Aberration",
        "Predicted Output Aberration",
    ],
):
    im = ax.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi)
    ax.set_title(title)
    ax.axis("off")
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])

plt.tight_layout()
plt.savefig("ml_results/phase_comparison.png", dpi=150)
print("Saved: ml_results/phase_comparison.png")

# Image reconstruction comparison
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Image Reconstruction", fontsize=14)
ax1.imshow(np.abs(ml_image), cmap="magma")
ax1.set_title("ML-Corrected Reconstruction")
ax1.axis("off")
ax2.imshow(np.abs(obj.obj), cmap="magma")
ax2.set_title("Ground Truth Object")
ax2.axis("off")

plt.tight_layout()
plt.savefig("ml_results/image_comparison.png", dpi=150)
print("Saved: ml_results/image_comparison.png")

plt.show()
