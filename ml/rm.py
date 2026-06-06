import torch


def RM_fft(M_rin_rout: torch.Tensor, N: int) -> torch.Tensor:
    M_temp = M_rin_rout.reshape(N, N, -1)
    M_temp = torch.fft.fftshift(M_temp, axes=(0, 1))
    M_temp = torch.fft.fft2(M_temp, axes=(0, 1))
    M_temp = torch.fft.ifftshift(M_temp, axes=(0, 1))

    M_temp = M_temp.reshape(N * N, -1).T

    M_temp = M_temp.reshape(N, N, -1)
    M_temp = torch.fft.fftshift(M_temp, axes=(0, 1))
    M_temp = torch.fft.fft2(M_temp, axes=(0, 1))
    M_temp = torch.fft.ifftshift(M_temp, axes=(0, 1))

    return M_temp.reshape(N * N, -1).T


def get_Rk(k_in: torch.Tensor, k_out: torch.Tensor, N: int) -> torch.Tensor:
    A = k_in.reshape(N * N, N * N)
    B = k_out.reshape(N * N, N * N)

    return B.T @ A.conj()
