"""
forward_pt.py
PyTorch-native forward simulation for the reflection-matrix microscope.

Key change from the original:  `simulate_pt_vectorized` now accepts an optional
`obj_batch` argument (shape: (B, 2N, 2N) complex).  When not provided it falls
back to sampling `batch_size` random objects from the scene pool via
`obj.get_batch_objs()`, so every training batch sees different objects.
"""

import torch
import torch.fft
import numpy as np

from config import config
from zern import generate_abberations
from obj import get_batch_objs
from forward_sim import c_in, c_out


def simulate_pt_vectorized(
    c_in_np=c_in,
    c_out_np=c_out,
    device="cpu",
    obj_batch: torch.Tensor | None = None,
    rng: np.random.Generator | None = None,
):
    """
    Batched forward simulation on GPU.

    Parameters
    ----------
    c_in_np  : (B, coeff_count) float32 array  — input aberration coefficients
    c_out_np : (B, coeff_count) float32 array  — output aberration coefficients
    device   : torch device string / object
    obj_batch: (B, 2N, 2N) complex tensor, or None.
               If None, B random scenes are sampled from the scene pool.
    rng      : optional numpy Generator for reproducible scene sampling.

    Returns
    -------
    k_ins  : (1, N, N, N, N) complex tensor  — input field (same for every sample)
    k_outs : (B, N, N, N, N) complex tensor  — output field per sample
    """
    N = config.N
    B = c_in_np.shape[0]

    # 1. Generate aberrations from Zernike coefficients (CPU → GPU)
    in_abb_batch  = np.stack([generate_abberations(c) for c in c_in_np])
    out_abb_batch = np.stack([generate_abberations(c) for c in c_out_np])

    input_abberations  = torch.tensor(in_abb_batch,  dtype=torch.cfloat, device=device)
    output_abberations = torch.tensor(out_abb_batch, dtype=torch.cfloat, device=device)

    # 2. Object images — sample B different scenes if not provided
    if obj_batch is None:
        obj_batch = get_batch_objs(B, rng=rng, device=device)  # (B, 2N, 2N)
    else:
        obj_batch = obj_batch.to(device=device, dtype=torch.cfloat)

    # 3. Build the base input grid: (N, N, 2N, 2N)
    #    Identical for all samples — only computed once per batch.
    k_in_base = torch.zeros((N, N, 2 * N, 2 * N), dtype=torch.cfloat, device=device)
    x_idx = torch.arange(N, device=device)[:, None]
    y_idx = torch.arange(N, device=device)
    k_in_base[x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0

    # k_ins is the same for every sample in the batch
    k_ins = k_in_base[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N].unsqueeze(0)

    # 4. Per-sample forward pass (sequential over batch to keep VRAM bounded)
    k_outs_list = []

    for b in range(B):
        obj_t = obj_batch[b]  # (2N, 2N)

        # Apply input aberrations in K-space
        s_inc_k = k_in_base * input_abberations[b]

        # K → R  (IFFT)
        s_inc_k_shifted = torch.fft.ifftshift(s_inc_k, dim=(-2, -1))
        s_inc_r_unshifted = torch.fft.ifft2(s_inc_k_shifted, dim=(-2, -1), norm="ortho")
        s_inc_r = torch.fft.fftshift(s_inc_r_unshifted, dim=(-2, -1))

        
        # Object interaction in real space (each k_in sees the SAME object)
        s_ref_r = s_inc_r * obj_t

        # R → K  (FFT)
        s_ref_r_shifted = torch.fft.ifftshift(s_ref_r, dim=(-2, -1))
        s_ref_k_unshifted = torch.fft.fft2(s_ref_r_shifted, dim=(-2, -1), norm="ortho")
        s_ref_k = torch.fft.fftshift(s_ref_k_unshifted, dim=(-2, -1))

        # Apply output aberrations and crop
        s_out_k = s_ref_k * output_abberations[b]
        k_outs_b = s_out_k[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        k_outs_list.append(k_outs_b)

    k_outs = torch.stack(k_outs_list, dim=0)  # (B, N, N, N, N)

    return k_ins, k_outs


# Single-sample version (unchanged from original)

def simulate_pt(c_in=c_in, c_out=c_out, device="cpu", obj_t=None):
    N = config.N

    in_abb_np  = generate_abberations(c_in)
    out_abb_np = generate_abberations(c_out)

    input_abberations  = torch.tensor(in_abb_np,  dtype=torch.cfloat, device=device)
    output_abberations = torch.tensor(out_abb_np, dtype=torch.cfloat, device=device)

    if obj_t is None:
        from obj import get_random_obj
        obj_t = get_random_obj(device=device, as_complex=True)

    k_outs = torch.zeros((N, N, N, N), dtype=torch.cfloat, device=device)
    k_ins  = torch.zeros((N, N, N, N), dtype=torch.cfloat, device=device)

    y_idx = torch.arange(N, device=device)

    for x in range(N):
        k_in_batch = torch.zeros((N, 2 * N, 2 * N), dtype=torch.cfloat, device=device)
        k_in_batch[y_idx, (N // 2) + x, (N // 2) + y_idx] = 1.0

        s_inc_k = k_in_batch * input_abberations
        s_inc_k_shifted = torch.fft.ifftshift(s_inc_k, dim=(-2, -1))
        s_inc_r_unshifted = torch.fft.ifft2(s_inc_k_shifted, dim=(-2, -1), norm="ortho")
        s_inc_r = torch.fft.fftshift(s_inc_r_unshifted, dim=(-2, -1))

        s_ref_r  = s_inc_r * obj_t
        s_ref_r_shifted = torch.fft.ifftshift(s_ref_r, dim=(-2, -1))
        s_ref_k_unshifted = torch.fft.fft2(s_ref_r_shifted, dim=(-2, -1), norm="ortho")
        s_ref_k = torch.fft.fftshift(s_ref_k_unshifted, dim=(-2, -1))
        
        s_out_k = s_ref_k * output_abberations

        k_ins[x, :]  = k_in_batch[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        k_outs[x, :] = s_out_k[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    return k_ins, k_outs
