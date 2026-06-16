# verified
import torch


def RM_fft(M_rin_rout: torch.Tensor, N: int) -> torch.Tensor:
    # 1. Process Output Coordinates (Rows)
    M_temp = M_rin_rout.reshape(N, N, -1)
    M_temp = torch.fft.fftshift(M_temp, dim=(0, 1))
    M_temp = torch.fft.fft2(M_temp, dim=(0, 1))
    M_temp = torch.fft.ifftshift(M_temp, dim=(0, 1))

    # 2. Transpose to switch focus to Input Coordinates (Columns)
    M_temp = M_temp.reshape(N * N, -1).T

    # 3. Process Input Coordinates
    M_temp = M_temp.reshape(N, N, -1)
    M_temp = torch.fft.fftshift(M_temp, dim=(0, 1))

    # Using inverse 2D FFT to mirror the original physical logic
    M_temp = torch.fft.ifft2(M_temp, dim=(0, 1))

    M_temp = torch.fft.ifftshift(M_temp, dim=(0, 1))

    # 4. Final Flatten and Transpose back to original orientation
    return M_temp.reshape(N * N, -1).T


def get_Rk(k_in: torch.Tensor, k_out: torch.Tensor, N: int) -> torch.Tensor:
    A = k_in.reshape(N * N, N * N)
    B = k_out.reshape(N * N, N * N)

    return B.T @ A.conj()


def get_Rk_batched(k_in: torch.Tensor, k_outs: torch.Tensor, N: int) -> torch.Tensor:
    """
    Computes the Reflection Matrix R_k for a batch of outputs against a single input.

    Expected shapes:
    k_in: (N, N, N, N)     -> Unbatched property from the simulator
    k_outs: (B, N, N, N, N) -> Batched output from the simulator
    """
    # 1. Reshape k_in to a 2D matrix: (N*N, N*N)
    A = k_in.reshape(N * N, N * N)

    # 2. Reshape k_outs to batched 2D matrices: (B, N*N, N*N)
    batch_size = k_outs.shape[0]
    B = k_outs.reshape(batch_size, N * N, N * N)

    # 3. Batched Matrix Multiplication
    # B.transpose(1, 2) swaps the last two dimensions of B -> shape: (B, N*N, N*N)
    # A.conj() -> shape: (N*N, N*N)
    # PyTorch broadcasting automatically handles: (B, N*N, N*N) @ (N*N, N*N)
    return B.transpose(1, 2) @ A.conj()
