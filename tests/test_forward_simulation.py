import numpy as np
import torch
import matplotlib.pyplot as plt
from config import config

# Cleaned up structural imports
from obj import obj as obj_np
from forward_sim import simulate_optimized, generate_abberations
from ml.forward import Simulation

# Fix random seeds for a reproducible comparison
np.random.seed(42)
torch.manual_seed(42)

# ==========================================
# 1. Prepare Mock System Inputs
# ==========================================
N = config.N

# Generate matching input and output aberrations
# Using arbitrary coefficients matching the design pattern
c_in = list(np.random.uniform(-0.5, 0.5, size=15))  # Adjusted to a standard size or nk
c_out = list(np.random.uniform(-0.5, 0.5, size=15))

ab_in_np = generate_abberations(c_in)
ab_out_np = generate_abberations(c_out)

# Convert arrays to PyTorch tensors for the ML simulation module
ab_in_pt = torch.tensor(ab_in_np, dtype=torch.complex64)
ab_out_pt = torch.tensor(ab_out_np, dtype=torch.complex64)
obj_pt = torch.tensor(obj_np, dtype=torch.complex64)

# ==========================================
# 2. Compute Outputs (NumPy vs PyTorch)
# ==========================================
# Run original NumPy implementation (using optimized batch version for fairness)
signal_in_np, signal_out_np = simulate_optimized(c_in=c_in, c_out=c_out)
k_ins_np = signal_in_np.k
k_outs_np = signal_out_np.k

# Initialize and run PyTorch Simulation Layer
sim_layer = Simulation(N=N, dtype=torch.complex64)
sim_layer.eval()

with torch.no_grad():
    k_ins_pt_tensor, k_outs_pt_tensor = sim_layer(ab_in_pt, ab_out_pt, obj_pt)
    k_ins_pt = k_ins_pt_tensor.cpu().numpy()
    k_outs_pt = k_outs_pt_tensor.cpu().numpy()

# ==========================================
# 3. Logical Comparison (Numerical Check)
# ==========================================
print("## LOGICAL COMPARISON RESULTS (K-SPACE) ##")
print("-" * 45)

# Calculate errors for Input K-Space Matrix
abs_diff_in = np.abs(k_ins_np - k_ins_pt)
max_diff_in = np.max(abs_diff_in)
mean_diff_in = np.mean(abs_diff_in)

# Calculate errors for Output K-Space Matrix
abs_diff_out = np.abs(k_outs_np - k_outs_pt)
max_diff_out = np.max(abs_diff_out)
mean_diff_out = np.mean(abs_diff_out)

print(f"INPUTS  -> Max Abs Diff: {max_diff_in:.2e} | Mean Abs Diff: {mean_diff_in:.2e}")
print(
    f"OUTPUTS -> Max Abs Diff: {max_diff_out:.2e} | Mean Abs Diff: {mean_diff_out:.2e}"
)

# Check if they match within typical float32 precision limits
# FFT normalization variations across frameworks may shift bounds slightly
is_matching_in = np.allclose(k_ins_np, k_ins_pt, atol=1e-5)
is_matching_out = np.allclose(k_outs_np, k_outs_pt, atol=1e-5)

print(f"Strict Input Match  (atol=1e-5): {is_matching_in}")
print(f"Strict Output Match (atol=1e-5): {is_matching_out}")
print("-" * 45)

# ==========================================
# 4. Visual Comparison (Plotting a Slice)
# ==========================================
# Since the arrays are 4D (N, N, N, N), we slice a central spatial frequency
# impulse point to inspect details visually.
mid_idx = N // 2

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- Row 1: k_outs Magnitude (NumPy vs PyTorch vs Delta Error) ---
im0 = axes[0, 0].imshow(np.abs(k_outs_np[mid_idx, mid_idx]), cmap="viridis")
axes[0, 0].set_title(f"NumPy k_out Magnitude (slice [{mid_idx},{mid_idx}])")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(np.abs(k_outs_pt[mid_idx, mid_idx]), cmap="viridis")
axes[0, 1].set_title(f"PyTorch k_out Magnitude (slice [{mid_idx},{mid_idx}])")
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[0, 2].imshow(abs_diff_out[mid_idx, mid_idx], cmap="inferno")
axes[0, 2].set_title("Absolute Magnitude Error")
fig.colorbar(im2, ax=axes[0, 2])

# --- Row 2: k_outs Phase (NumPy vs PyTorch vs Delta Error) ---
im3 = axes[1, 0].imshow(
    np.angle(k_outs_np[mid_idx, mid_idx]), cmap="twilight", vmin=-np.pi, vmax=np.pi
)
axes[1, 0].set_title("NumPy k_out Phase")
fig.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(
    np.angle(k_outs_pt[mid_idx, mid_idx]), cmap="twilight", vmin=-np.pi, vmax=np.pi
)
axes[1, 1].set_title("PyTorch k_out Phase")
fig.colorbar(im4, ax=axes[1, 1])

# Isolate phase difference calculation where values are significant
magnitude_mask = np.abs(k_outs_np[mid_idx, mid_idx]) > 1e-5
phase_diff = np.zeros((N, N))
phase_diff[magnitude_mask] = np.angle(
    k_outs_np[mid_idx, mid_idx][magnitude_mask]
) - np.angle(k_outs_pt[mid_idx, mid_idx][magnitude_mask])

im5 = axes[1, 2].imshow(phase_diff, cmap="inferno")
axes[1, 2].set_title("Phase Error (Masked)")
fig.colorbar(im5, ax=axes[1, 2])

plt.suptitle(f"Forward Pass Vectorized Verification (Grid Size: N={N})", fontsize=16)
plt.tight_layout()
plt.show()
