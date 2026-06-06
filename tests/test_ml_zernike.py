import numpy as np
import torch
import matplotlib.pyplot as plt
from config import config

# Cleaned up structural imports as requested
from zern import generate_abberations as generate_aberrations_numpy
from ml.zern import ZernikeAberration

# Fix random seeds for a reproducible comparison
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Initialize PyTorch Layer & Generate Coefficients
# ==========================================
aberration_layer = ZernikeAberration(N=config.N, zern_n=config.zern_n)
aberration_layer.eval()
nk = aberration_layer.num_coefficients

# Generate matching test coefficients for both frameworks
test_coeffs_np = np.random.uniform(-1.5, 1.5, size=nk)
test_coeffs_pt = torch.tensor(test_coeffs_np, dtype=torch.float32)

# ==========================================
# 2. Compute Outputs
# ==========================================
# Run original NumPy implementation from zern.py
output_np = generate_aberrations_numpy(test_coeffs_np)

# Run new PyTorch implementation from ml/zern.py (on CPU for parity evaluation)
with torch.no_grad():
    output_pt = aberration_layer(test_coeffs_pt).cpu().numpy()

# ==========================================
# 3. Logical Comparison (Numerical Check)
# ==========================================
print("## LOGICAL COMPARISON RESULTS ##")
print("-" * 32)

abs_diff = np.abs(output_np - output_pt)
max_diff = np.max(abs_diff)
mean_diff = np.mean(abs_diff)

print(f"Maximum Absolute Difference: {max_diff:.2e}")
print(f"Mean Absolute Difference:    {mean_diff:.2e}")

# Check if they match within standard float32 precision limits (1e-6)
is_matching = np.allclose(output_np, output_pt, atol=1e-6)
print(f"Strict Numerical Match (atol=1e-6): {is_matching}")
print("-" * 32)

# ==========================================
# 4. Visual Comparison (Plotting)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- Row 1: Amplitude (Magnitude) ---
im0 = axes[0, 0].imshow(np.abs(output_np), cmap="gray", origin="lower")
axes[0, 0].set_title("NumPy Amplitude (zern.py)")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(np.abs(output_pt), cmap="gray", origin="lower")
axes[0, 1].set_title("PyTorch Amplitude (ml/zern.py)")
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(np.abs(output_np - output_pt), cmap="inferno", origin="lower")
axes[0, 2].set_title("Amplitude Difference Error")
fig.colorbar(im2, ax=axes[0, 2])

# --- Row 2: Phase (Angle) ---
im3 = axes[1, 0].imshow(np.angle(output_np), cmap="jet", origin="lower")
axes[1, 0].set_title("NumPy Phase (zern.py)")
fig.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(np.angle(output_pt), cmap="jet", origin="lower")
axes[1, 1].set_title("PyTorch Phase (ml/zern.py)")
fig.colorbar(im4, ax=axes[1, 1])

# Isolate phase difference calculation to the active pupil mask area
pupil_mask = np.abs(output_np) > 0
phase_diff = np.zeros_like(output_np, dtype=np.float64)
phase_diff[pupil_mask] = np.angle(output_np[pupil_mask]) - np.angle(
    output_pt[pupil_mask]
)

im5 = axes[1, 2].imshow(phase_diff, cmap="inferno", origin="lower")
axes[1, 2].set_title("Phase Difference Error")
fig.colorbar(im5, ax=axes[1, 2])

plt.suptitle(
    f"Zernike Aberration Verification (N={config.N}, Max Order={config.zern_n})",
    fontsize=16,
)
plt.tight_layout()
plt.show()
