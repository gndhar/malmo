import numpy as np
import torch
import matplotlib.pyplot as plt
from config import config

# Direct imports from production code
from reflection_matrix import RM_fft as rm_fft_np, generate_R_k as get_rk_np
from ml.rm import RM_fft as rm_fft_pt, get_Rk as get_rk_pt
from forward_sim import Signal, Space

# Seed everything for deterministic runs
np.random.seed(42)
torch.manual_seed(42)

## =====================================================================
## 1. MOCK DATA SETUPS
## =====================================================================
N = config.N
matrix_dim = N * N

# Generate native NumPy complex128 mock matrices
kin_np = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
kout_np = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
rm_np = np.random.randn(matrix_dim, matrix_dim) + 1j * np.random.randn(
    matrix_dim, matrix_dim
)

# Wrap NumPy representations inside the production framework abstraction
sig_in = Signal(kin_np, Space.K)
sig_out = Signal(kout_np, Space.K)

# Convert to PyTorch tensors across both potential target precisions
# (Catches implementation deviations driven purely by float32 vs float64 limits)
kin_pt_c64 = torch.tensor(kin_np, dtype=torch.complex64)
kout_pt_c64 = torch.tensor(kout_np, dtype=torch.complex64)
rm_pt_c64 = torch.tensor(rm_np, dtype=torch.complex64)

kin_pt_c128 = torch.tensor(kin_np, dtype=torch.complex128)
kout_pt_c128 = torch.tensor(kout_np, dtype=torch.complex128)
rm_pt_c128 = torch.tensor(rm_np, dtype=torch.complex128)


## =====================================================================
## 2. EXECUTE GENERATION OPERATIONS (R_k)
## =====================================================================
print("## VERIFYING R_k GENERATION PIPELINE ##")
print("=" * 65)

# Run raw production pipelines
R_k_np = get_rk_np(sig_in, sig_out)
R_k_pt_c64 = get_rk_pt(kin_pt_c64, kout_pt_c64, N).cpu().numpy()
R_k_pt_c128 = get_rk_pt(kin_pt_c128, kout_pt_c128, N).cpu().numpy()

# Analyze precision vs algorithmic correctness
err_rk_c64 = np.max(np.abs(R_k_np - R_k_pt_c64))
err_rk_c128 = np.max(np.abs(R_k_np - R_k_pt_c128))

print(f"Max absolute delta (FP32/Complex64 Backend):  {err_rk_c64:.2e}")
print(f"Max absolute delta (FP64/Complex128 Backend): {err_rk_c128:.2e}")

match_rk_64 = np.allclose(R_k_np, R_k_pt_c64, atol=1e-3)
match_rk_128 = np.allclose(R_k_np, R_k_pt_c128, atol=1e-5)

print(
    f"Mathematical equivalence check (Complex64 at 1e-3):  {'PASS' if match_rk_64 else 'FAIL'}"
)
print(
    f"Mathematical equivalence check (Complex128 at 1e-5): {'PASS' if match_rk_128 else 'FAIL'}"
)
print("=" * 65 + "\n")


## =====================================================================
## 3. EXECUTE TRANSFORM OPERATIONS (RM_fft)
## =====================================================================
print("## VERIFYING RM_fft TRANSFORMATION PIPELINE ##")
print("=" * 65)

# Safely call production code to isolate syntax crashes from mathematical logic bugs
R_r_np = None
R_r_pt = None
rm_fft_crashed = False

try:
    R_r_np = rm_fft_np(rm_np)
except Exception as e:
    print(
        f"[CRITICAL FAILURE] NumPy production function `RM_fft` threw an error:\n-> {type(e).__name__}: {e}"
    )
    rm_fft_crashed = True

try:
    # Testing against complex64 as it matches standard PyTorch layer instantiation conventions
    R_r_pt = rm_fft_pt(rm_pt_c64, N).cpu().numpy()
except Exception as e:
    print(
        f"[CRITICAL FAILURE] PyTorch production function `RM_fft` threw an error:\n-> {type(e).__name__}: {e}"
    )
    rm_fft_crashed = True

if not rm_fft_crashed:
    err_rm_fft = np.max(np.abs(R_r_np - R_r_pt))
    match_rm = np.allclose(R_r_np, R_r_pt, atol=1e-4)
    print(f"Max absolute delta between pipelines: {err_rm_fft:.2e}")
    print(f"Mathematical equivalence check (at 1e-4): {'PASS' if match_rm else 'FAIL'}")
else:
    print(
        "[BYPASSED] Numerical parity check skipped because production functions failed to execute."
    )
print("=" * 65 + "\n")


## =====================================================================
## 4. ISOLATED VISUAL DIAGNOSTICS (No Complex128/Matplotlib Crashes)
## =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# --- Row 1: R_k Pipeline Diagnosis ---
axes[0, 0].imshow(np.abs(R_k_np), cmap="viridis")
axes[0, 0].set_title("NumPy R_k Magnitude (Complex128)")

axes[0, 1].imshow(np.abs(R_k_pt_c128), cmap="viridis")
axes[0, 1].set_title("PyTorch R_k Magnitude (Complex128)")

# Render difference cleanly as real-valued float64 magnitudes
im_rk_err = axes[0, 2].imshow(np.abs(R_k_np - R_k_pt_c128), cmap="inferno")
axes[0, 2].set_title("R_k Absolute Errors (FP64 Matrix)")
fig.colorbar(im_rk_err, ax=axes[0, 2])

# --- Row 2: RM_fft Pipeline Diagnosis ---
if not rm_fft_crashed and R_r_np is not None and R_r_pt is not None:
    axes[1, 0].imshow(np.abs(R_r_np), cmap="viridis")
    axes[1, 1].imshow(np.abs(R_r_pt), cmap="viridis")
    im_rm_err = axes[1, 2].imshow(np.abs(R_r_np - R_r_pt), cmap="inferno")
    fig.colorbar(im_rm_err, ax=axes[1, 2])
else:
    # Explicit zero fallback explicitly cast to pure float to protect Matplotlib imshow
    blank_matrix = np.zeros((matrix_dim, matrix_dim), dtype=np.float64)
    axes[1, 0].imshow(blank_matrix, cmap="gray")
    axes[1, 1].imshow(blank_matrix, cmap="gray")
    axes[1, 2].imshow(blank_matrix, cmap="gray")
    axes[1, 2].text(
        0.5,
        0.5,
        "EXECUTION CRASHED\nSEE TERMINAL LOGS",
        color="white",
        ha="center",
        va="center",
        transform=axes[1, 2].transAxes,
    )

axes[1, 0].set_title("NumPy RM_fft Magnitude")
axes[1, 1].set_title("PyTorch RM_fft Magnitude")
axes[1, 2].set_title("RM_fft Absolute Errors")

plt.suptitle(f"Parity Verification Suite (System Dimension N={N})", fontsize=16)
plt.tight_layout()
plt.show()
