import numpy as np
import torch

# Import your configurations and modules
from config import config
import ml.rm as rm_pt
import reflection_matrix as rm_np


# Mock Signal class to replicate forward_sim.Signal behavior for testing
class MockSignal:

    def __init__(self, data: np.ndarray):
        self.k = data
        self.r = np.fft.ifft2(data)  # Just mock the real-space representation if needed


def test_rm_functions():
    N = config.N
    print(f"Testing Reflection Matrix operations for N={N}...\n")

    # Use complex128 for exact precision matching
    pt_dtype = torch.complex128
    np_dtype = np.complex128

    # ==========================================
    # 1. Test RM_fft (Real to K-space / K to Real mapping)
    # ==========================================
    print("--- Testing RM_fft ---")

    # Generate random matrix of shape (N*N, N*N)
    M_dummy_np = np.random.randn(N * N, N * N) + 1j * np.random.randn(N * N, N * N)
    M_dummy_np = M_dummy_np.astype(np_dtype)
    M_dummy_pt = torch.from_numpy(M_dummy_np).to(pt_dtype)

    # Run both implementations
    M_fft_np = rm_np.RM_fft(M_dummy_np)
    M_fft_pt_tensor = rm_pt.RM_fft(M_dummy_pt, N)
    M_fft_pt = M_fft_pt_tensor.numpy()

    # Compare
    max_err_fft = np.max(np.abs(M_fft_np - M_fft_pt))
    print(f"RM_fft Maximum absolute error: {max_err_fft:.2e}")
    np.testing.assert_allclose(M_fft_np, M_fft_pt, atol=1e-11, rtol=1e-11)
    print("✅ RM_fft outputs match perfectly!")

    # ==========================================
    # 1B. Test RM_ifft Round-trip Inversion (Assuming Even N)
    # ==========================================
    print("\n--- Testing RM_ifft Round-trip ---")
    assert N % 2 == 0, f"N must be even for this test suite, but got N={N}"

    # Direction 1: Real-space -> K-space (RM_fft) -> Real-space (RM_ifft)
    M_rec_pt = rm_pt.RM_ifft(M_fft_pt_tensor, N)
    max_err_ifft = torch.max(torch.abs(M_dummy_pt - M_rec_pt)).item()
    print(f"RM_ifft(RM_fft(M_space)) max absolute error: {max_err_ifft:.2e}")
    np.testing.assert_allclose(M_dummy_np, M_rec_pt.numpy(), atol=1e-11, rtol=1e-11)

    # Direction 2: K-space -> Real-space (RM_ifft) -> K-space (RM_fft)
    # Generate random k-space matrix directly
    M_k_dummy = (
        np.random.randn(N * N, N * N) + 1j * np.random.randn(N * N, N * N)
    ).astype(np_dtype)
    M_k_pt = torch.from_numpy(M_k_dummy).to(pt_dtype)

    M_space_pt = rm_pt.RM_ifft(M_k_pt, N)
    M_k_rec_pt = rm_pt.RM_fft(M_space_pt, N)

    max_err_k_roundtrip = torch.max(torch.abs(M_k_pt - M_k_rec_pt)).item()
    print(f"RM_fft(RM_ifft(M_kspace)) max absolute error: {max_err_k_roundtrip:.2e}")
    np.testing.assert_allclose(M_k_dummy, M_k_rec_pt.numpy(), atol=1e-11, rtol=1e-11)

    print("✅ RM_ifft perfectly inverts RM_fft in both directions!")
    # ==========================================
    # 2. Test get_Rk / generate_R_k (Single Input/Output R_k Generation)
    # ==========================================
    print("\n--- Testing get_Rk vs generate_R_k ---")

    # Generate random k_in and k_out matrices of shape (N, N, N, N)
    k_in_np = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)
    k_out_np = np.random.randn(N, N, N, N) + 1j * np.random.randn(N, N, N, N)

    k_in_np = k_in_np.astype(np_dtype)
    k_out_np = k_out_np.astype(np_dtype)

    # Convert to PyTorch tensors and NumPy mock Signals
    k_in_pt = torch.from_numpy(k_in_np).to(pt_dtype)
    k_out_pt = torch.from_numpy(k_out_np).to(pt_dtype)

    sig_in = MockSignal(k_in_np)
    sig_out = MockSignal(k_out_np)

    # Run both implementations
    Rk_np = rm_np.generate_R_k(sig_in, sig_out)
    Rk_pt = rm_pt.get_Rk(k_in_pt, k_out_pt, N).numpy()

    # Compare
    max_err_rk = np.max(np.abs(Rk_np - Rk_pt))
    print(f"get_Rk Maximum absolute error: {max_err_rk:.2e}")
    np.testing.assert_allclose(Rk_np, Rk_pt, atol=1e-11, rtol=1e-11)
    print("✅ get_Rk outputs match perfectly!")

    # ==========================================
    # 3. Test get_Rk_batched (Batched PyTorch against Looped NumPy)
    # ==========================================
    print("\n--- Testing get_Rk_batched ---")

    batch_size = 4

    # k_in is identical for all items in the batch (shape: N, N, N, N)
    # k_outs is batched (shape: B, N, N, N, N)
    k_outs_batched_np = np.random.randn(batch_size, N, N, N, N) + 1j * np.random.randn(
        batch_size, N, N, N, N
    )
    k_outs_batched_np = k_outs_batched_np.astype(np_dtype)

    k_outs_batched_pt = torch.from_numpy(k_outs_batched_np).to(pt_dtype)

    # 3A. Run Batched PyTorch
    Rk_batched_pt = rm_pt.get_Rk_batched(k_in_pt, k_outs_batched_pt, N).numpy()

    # 3B. Run Looped NumPy
    Rk_batched_np = np.zeros((batch_size, N * N, N * N), dtype=np_dtype)
    for b in range(batch_size):
        sig_out_b = MockSignal(k_outs_batched_np[b])
        Rk_batched_np[b] = rm_np.generate_R_k(sig_in, sig_out_b)

    # Compare shapes and data
    assert Rk_batched_np.shape == Rk_batched_pt.shape, "Batched shapes do not match!"
    max_err_batched = np.max(np.abs(Rk_batched_np - Rk_batched_pt))
    print(f"get_Rk_batched Maximum absolute error: {max_err_batched:.2e}")
    np.testing.assert_allclose(Rk_batched_np, Rk_batched_pt, atol=1e-11, rtol=1e-11)
    print("✅ get_Rk_batched outputs match perfectly!")

    print("\n🎉 All Reflection Matrix tests passed!")


if __name__ == "__main__":
    test_rm_functions()
