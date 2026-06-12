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
