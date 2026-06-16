import torch
import numpy as np

# Import from your numpy implementation
import forward_sim as fwd_np
from config import config

# Import the PyTorch module
from ml.forward import Simulation as SimulationPT


def test_simulation_equivalence():
    N = config.N
    print(f"Testing Simulation equivalence for N={N}...")

    # ==========================================
    # 1. Run NumPy Simulation
    # ==========================================
    print("Running NumPy simulate()...")
    # Using simulate_optimized or simulate, since they yield the same result
    _, s_out_np = fwd_np.simulate()
    k_outs_np = s_out_np.k

    # ==========================================
    # 2. Setup PyTorch Simulation
    # ==========================================
    print("Running PyTorch Simulation...")

    # Initialize module (CPU is fine for testing equivalence)
    sim_pt = SimulationPT(N=N, dtype=torch.complex128)
    sim_pt.eval()

    # Convert NumPy arrays to PyTorch Tensors
    # We use exactly what was generated globally in forward.py
    ab_in_pt = torch.from_numpy(fwd_np.input_abberations).to(torch.complex64)
    ab_out_pt = torch.from_numpy(fwd_np.output_abberations).to(torch.complex64)
    obj_pt = torch.from_numpy(fwd_np.obj).to(torch.complex64)

    # ==========================================
    # 3. Run PyTorch Forward Pass
    # ==========================================
    with torch.no_grad():
        k_outs_pt = sim_pt(ab_in_pt, ab_out_pt, obj_pt)

    # Convert output back to NumPy for comparison
    k_outs_pt_np = k_outs_pt.cpu().numpy()

    # ==========================================
    # 4. Compare Outputs
    # ==========================================
    print("\nComparing outputs...")

    # Check shape
    assert (
        k_outs_np.shape == k_outs_pt_np.shape
    ), f"Shape mismatch: NumPy {k_outs_np.shape} vs PyTorch {k_outs_pt_np.shape}"

    # Calculate absolute and relative errors
    abs_diff = np.abs(k_outs_np - k_outs_pt_np)
    max_abs_err = np.max(abs_diff)

    print(f"Maximum absolute error: {max_abs_err:.2e}")

    try:
        # We use a slight tolerance to account for standard float/complex precision differences
        # between numpy CPU and torch CPU backend FFT algorithms.
        np.testing.assert_allclose(k_outs_np, k_outs_pt_np, rtol=1e-4, atol=1e-5)
        print("✅ SUCCESS: PyTorch Simulation matches NumPy Simulation!")
    except AssertionError as e:
        print("❌ FAILURE: Outputs do not match.")
        print(e)

        # Note on FFT Normalization:
        print(
            "\nNote: If the outputs are off by a massive constant scalar, check your custom "
            "`fft.py` file. PyTorch is using `norm='ortho'`, so if your numpy `fft2` uses "
            "the default (backward) normalization, you will need to apply `norm='ortho'` "
            "in your numpy implementation to get identical values."
        )


if __name__ == "__main__":
    test_simulation_equivalence()
