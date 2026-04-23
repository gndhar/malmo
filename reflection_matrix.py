from forward_sim import Signal
import numpy as np
from config import config

N = config.N


def RM_fft(M_xin_xout):
    Nx = Ny = config.N
    if M_xin_xout.shape[0] != Nx * Ny:
        raise ValueError("Check matrix dimensions!!")

    # 2. Process Output Coordinates (Rows)
    # Reshape using Fortran order to match MATLAB's (Ny, Nx, [])
    # -1 tells NumPy to calculate the remaining dimension automatically
    M_temp = np.reshape(M_xin_xout, (Ny, Nx, -1))

    # Apply fftshift on the first two axes (spatial dimensions)
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 2D FFT on the first two axes
    M_temp = np.fft.fft2(M_temp, axes=(0, 1))

    # Shift back
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 3. Transpose to switch focus to Input Coordinates (Columns)
    # Reshape to (Ny*Nx, []) and Transpose
    M_temp = np.reshape(M_temp, (Ny * Nx, -1)).T

    # 4. Process Input Coordinates
    M_temp = np.reshape(M_temp, (Ny, Nx, -1))
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # Note: Using IFFT2 as per the original MATLAB logic
    M_temp = np.fft.ifft2(M_temp, axes=(0, 1))

    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 5. Final Flatten and Transpose back to original orientation
    M_kin_kout = np.reshape(M_temp, (Ny * Nx, -1)).T

    return M_kin_kout


def generate_R_k(s_in: Signal, s_out: Signal):
    N = config.N

    # 1. Reshape creates an instant "View" (Zero memory copying)
    # V_A shape: (N^2, N^2), where each row is a flattened N x N input
    V_A = s_in.k.reshape(N * N, N * N)
    V_B = s_out.k.reshape(N * N, N * N)

    # 2. B_k is the transpose of V_B.
    # A_k.conj().T algebraically simplifies to just V_A.conj()
    R_k = V_B.T @ V_A.conj()

    return R_k


def generate_R(s_in: Signal, s_out: Signal):
    N = config.N

    # K-Space
    V_A_k = s_in.k.reshape(N * N, N * N)
    V_B_k = s_out.k.reshape(N * N, N * N)
    R_k = V_B_k.T @ V_A_k.conj()

    # Real Space
    V_A_r = s_in.r.reshape(N * N, N * N)
    V_B_r = s_out.r.reshape(N * N, N * N)
    R = V_B_r.T @ V_A_r.conj()

    # If you still need A and B explicitly returned as columns for other code:
    A = V_A_r.T
    B = V_B_r.T

    return R, A, B, R_k


import torch


def generate_R_k_pt(k_ins: torch.Tensor, k_outs: torch.Tensor):
    N = config.N

    # 1. Reshape creates zero-copy memory views on the GPU
    # Shape becomes (N^2, N^2)
    V_A = k_ins.reshape(N * N, N * N)
    V_B = k_outs.reshape(N * N, N * N)

    # 2. Hardware-accelerated complex matrix multiplication
    R_k = V_B.T @ V_A.conj()

    return R_k
