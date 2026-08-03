import numpy as np
import forward_sim
import torch

from ml.zern import ZernikeAberration
from ml.forward import Simulation
from ml.rm import get_Rk_batched
from reflection_matrix import generate_R_k

import matplotlib.pyplot as plt

# Global setup using N = 32
N = 32
zern_gen = ZernikeAberration(N=N, zern_n=5, dtype=torch.float64, npdtype=np.float64)


def pytorch_forward(tc_in, tc_out):
    simulation = Simulation(N=N, dtype=torch.complex128)

    ab_in = zern_gen(tc_in)
    ab_out = zern_gen(tc_out)

    print("ab_in / ab_out shapes:", ab_in.shape, ab_out.shape)

    # 2N x 2N spatial grid = 64 x 64
    obj = torch.tensor(forward_sim.obj, dtype=torch.complex128).reshape(1, 2 * N, 2 * N)
    print("forward_sim.obj shape:", forward_sim.obj.shape)
    print("Reshaped obj tensor shape:", obj.shape)

    k_outs = simulation(ab_in, ab_out, obj)

    print(
        "k_outs shape / k_in_cropped shape:",
        k_outs.shape,
        simulation.k_in_cropped.shape,
    )

    Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)
    return Rk, ab_in, ab_out


def test_batch_leak():
    x = 4  # Base batch size

    # 1. Pass Sample Set 1 (size N = x)
    tc_in1 = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)
    tc_out1 = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)
    Rk1, *_ = pytorch_forward(tc_in1, tc_out1)

    # 2. Pass Sample Set 2 (size N = x)
    tc_in2 = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)
    tc_out2 = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)
    Rk2, *_ = pytorch_forward(tc_in2, tc_out2)

    # 3. Concatenate sample inputs into a single batch (size N = 2x)
    tc_in_combined = torch.cat([tc_in1, tc_in2], dim=0)
    tc_out_combined = torch.cat([tc_out1, tc_out2], dim=0)

    Rk_combined, *_ = pytorch_forward(tc_in_combined, tc_out_combined)

    # 4. Concatenate individual outputs along batch dimension (dim 0)
    Rk_expected = torch.cat([Rk1, Rk2], dim=0)

    # 5. Verify batch consistency within float64 precision tolerance
    assert torch.equal(Rk_combined, Rk_expected)
    print("Success: Batch processing (N=2x) matches individual passes (2 x N=x)!")


def test_forward():
    x = 1
    tc_in = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)
    tc_out = torch.randn(x, zern_gen.num_coefficients, dtype=torch.float64)

    # 1. PyTorch Forward Pass
    Rk, ab_in, ab_out = pytorch_forward(tc_in, tc_out)
    Rk = Rk.reshape(N, N, N, N)  # Reshape to (32, 32, 32, 32)

    c_in = tc_in.flatten().tolist()
    c_out = tc_out.flatten().tolist()

    # 2. NumPy Reference Calculations
    ab_in_np_raw = forward_sim.generate_abberations(c_in)
    ab_out_np_raw = forward_sim.generate_abberations(c_out)
    s_in, s_out = forward_sim.simulate_optimized(c_in, c_out)
    R_k_raw = generate_R_k(s_in, s_out)

    # 3. Convert to Tensors
    ab_in_pt = ab_in.squeeze(0)
    ab_out_pt = ab_out.squeeze(0)

    ab_in_np = torch.tensor(ab_in_np_raw, dtype=torch.complex128)
    ab_out_np = torch.tensor(ab_out_np_raw, dtype=torch.complex128)
    R_k_np = torch.tensor(R_k_raw, dtype=torch.complex128).reshape(N, N, N, N)

    # 4. Numerical Assertions
    torch.testing.assert_close(ab_in_pt, ab_in_np, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(ab_out_pt, ab_out_np, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(Rk, R_k_np, rtol=1e-10, atol=1e-10)
    print("✅ All numerical assertions passed! PyTorch matches NumPy.")

    # 5. Extract Phase & Central Mode Slices (N // 2 = 16)
    center_idx = N // 2
    phase_in_pt = torch.angle(ab_in_pt).cpu().numpy()
    phase_in_np = torch.angle(ab_in_np).cpu().numpy()

    phase_out_pt = torch.angle(ab_out_pt).cpu().numpy()
    phase_out_np = torch.angle(ab_out_np).cpu().numpy()

    # Slice central spatial mode of Reflection Matrix
    rk_pt_slice = torch.abs(Rk[center_idx, center_idx]).cpu().numpy()
    rk_np_slice = torch.abs(R_k_np[center_idx, center_idx]).cpu().numpy()

    # 6. Plotting Grid (3x3)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))

    # --- Row 1: ab_in Phase ---
    im0 = axes[0, 0].imshow(phase_in_pt, cmap="twilight")
    axes[0, 0].set_title("ab_in Phase (PyTorch)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(phase_in_np, cmap="twilight")
    axes[0, 1].set_title("ab_in Phase (NumPy)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(np.abs(phase_in_pt - phase_in_np), cmap="inferno")
    axes[0, 2].set_title("ab_in Abs Difference")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Row 2: ab_out Phase ---
    im3 = axes[1, 0].imshow(phase_out_pt, cmap="twilight")
    axes[1, 0].set_title("ab_out Phase (PyTorch)")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(phase_out_np, cmap="twilight")
    axes[1, 1].set_title("ab_out Phase (NumPy)")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(np.abs(phase_out_pt - phase_out_np), cmap="inferno")
    axes[1, 2].set_title("ab_out Abs Difference")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # --- Row 3: Reflection Matrix Slice Magnitude ---
    im6 = axes[2, 0].imshow(rk_pt_slice, cmap="viridis")
    axes[2, 0].set_title(f"|Rk[{center_idx},{center_idx}]| (PyTorch)")
    fig.colorbar(im6, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im7 = axes[2, 1].imshow(rk_np_slice, cmap="viridis")
    axes[2, 1].set_title(f"|Rk[{center_idx},{center_idx}]| (NumPy)")
    fig.colorbar(im7, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im8 = axes[2, 2].imshow(np.abs(rk_pt_slice - rk_np_slice), cmap="inferno")
    axes[2, 2].set_title("Rk Slice Abs Difference")
    fig.colorbar(im8, ax=axes[2, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("PyTorch vs NumPy Simulation Verification (N=32)", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_batch_leak()
    test_forward()
