import torch
import torch.fft
import numpy as np

from config import config
from zern import generate_abberations
from obj import obj  # Assuming this is your numpy object array
from forward_sim import c_in, c_out


def simulate_pt_vectorized(c_in_np=c_in, c_out_np=c_out, device="cpu"):
    N = config.N
    batch_size = c_in_np.shape[0]

    # 1. Generate aberrations in NumPy (loop over the batch on CPU)
    in_abb_batch = np.stack([generate_abberations(c) for c in c_in_np])
    out_abb_batch = np.stack([generate_abberations(c) for c in c_out_np])

    # Move batch to GPU and reshape for 5D broadcasting: (B, 1, 1, 2N, 2N)
    input_abberations = torch.tensor(
        in_abb_batch, dtype=torch.cfloat, device=device
    ).view(batch_size, 1, 1, 2 * N, 2 * N)
    output_abberations = torch.tensor(
        out_abb_batch, dtype=torch.cfloat, device=device
    ).view(batch_size, 1, 1, 2 * N, 2 * N)

    # Assuming obj is a 2D numpy array of shape (2N, 2N)
    obj_t = torch.tensor(obj, dtype=torch.cfloat, device=device).view(
        1, 1, 1, 2 * N, 2 * N
    )

    # 2. Vectorized Input Grid: Shape (1, N, N, 2N, 2N)
    k_in_batch = torch.zeros((1, N, N, 2 * N, 2 * N), dtype=torch.cfloat, device=device)
    x_idx = torch.arange(N, device=device)[:, None]
    y_idx = torch.arange(N, device=device)
    k_in_batch[0, x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0

    # 3. Apply Aberrations: Result automatically broadcasts to (B, N, N, 2N, 2N)
    s_inc_k = k_in_batch * input_abberations

    # 4. Transform K -> R Space
    s_inc_k_shifted = torch.fft.ifftshift(s_inc_k, dim=(-2, -1))
    s_inc_r_unshifted = torch.fft.ifft2(s_inc_k_shifted, dim=(-2, -1), norm="ortho")
    s_inc_r = torch.fft.fftshift(s_inc_r_unshifted, dim=(-2, -1))

    # 5. Object Interaction
    s_ref_r = s_inc_r * obj_t

    # 6. Transform R -> K Space
    s_ref_r_shifted = torch.fft.ifftshift(s_ref_r, dim=(-2, -1))
    s_ref_k_unshifted = torch.fft.fft2(s_ref_r_shifted, dim=(-2, -1), norm="ortho")
    s_ref_k = torch.fft.fftshift(s_ref_k_unshifted, dim=(-2, -1))

    # 7. Apply Output Aberrations
    s_out_k = s_ref_k * output_abberations

    # 8. Crop to N x N
    # k_ins remains shape (1, N, N, N, N) because it is identical for every sample in the batch!
    k_ins = k_in_batch[:, :, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
    # k_outs takes on the full batch size: (B, N, N, N, N)
    k_outs = s_out_k[:, :, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    return k_ins, k_outs


import torch
import torch.fft
import numpy as np

from config import config
from zern import generate_abberations
from obj import obj


def simulate_pt_vectorized(c_in_np=c_in, c_out_np=c_out, device="cpu"):
    N = config.N
    B = c_in_np.shape[0]

    # 1. Generate aberrations in NumPy
    in_abb_batch = np.stack([generate_abberations(c) for c in c_in_np])
    out_abb_batch = np.stack([generate_abberations(c) for c in c_out_np])

    # Move directly to GPU (Shape: B, 2N, 2N)
    input_abberations = torch.tensor(in_abb_batch, dtype=torch.cfloat, device=device)
    output_abberations = torch.tensor(out_abb_batch, dtype=torch.cfloat, device=device)
    obj_t = torch.tensor(obj, dtype=torch.cfloat, device=device)

    # 2. Vectorized Base Input Grid: Shape (N, N, 2N, 2N)
    # This grid is identical for every sample, so we only make it once.
    k_in_base = torch.zeros((N, N, 2 * N, 2 * N), dtype=torch.cfloat, device=device)
    x_idx = torch.arange(N, device=device)[:, None]
    y_idx = torch.arange(N, device=device)
    k_in_base[x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0

    # k_ins never changes based on aberrations, so we crop it once.
    # Add a dummy batch dimension so it returns as (1, N, N, N, N)
    k_ins = k_in_base[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N].unsqueeze(0)

    # 3. Process the batch sequentially to save VRAM
    k_outs_list = []

    for b in range(B):
        # Apply Aberrations for this specific sample
        # Broadcasting here only takes ~536 MB
        s_inc_k = k_in_base * input_abberations[b]

        # 4. Transform K -> R Space
        s_inc_k_shifted = torch.fft.ifftshift(s_inc_k, dim=(-2, -1))
        s_inc_r_unshifted = torch.fft.ifft2(s_inc_k_shifted, dim=(-2, -1), norm="ortho")
        s_inc_r = torch.fft.fftshift(s_inc_r_unshifted, dim=(-2, -1))

        # 5. Object Interaction
        s_ref_r = s_inc_r * obj_t

        # 6. Transform R -> K Space
        s_ref_r_shifted = torch.fft.ifftshift(s_ref_r, dim=(-2, -1))
        s_ref_k_unshifted = torch.fft.fft2(s_ref_r_shifted, dim=(-2, -1), norm="ortho")
        s_ref_k = torch.fft.fftshift(s_ref_k_unshifted, dim=(-2, -1))

        # 7. Apply Output Aberrations
        s_out_k = s_ref_k * output_abberations[b]

        # 8. Crop to N x N and append
        k_outs_b = s_out_k[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        k_outs_list.append(k_outs_b)

    # 9. Stack the outputs back into a single batch tensor: (B, N, N, N, N)
    k_outs = torch.stack(k_outs_list, dim=0)

    return k_ins, k_outs


# previous version
import torch
import torch.fft
import numpy as np

from config import config
from zern import generate_abberations
from obj import obj  # Assuming this is your numpy object array


def simulate_pt(c_in=c_in, c_out=c_out, device="cpu"):
    N = config.N

    # 1. Generate aberrations (NumPy to PyTorch)
    # We generate on CPU (if zern is pure numpy) and immediately move to GPU
    in_abb_np = generate_abberations(c_in)
    out_abb_np = generate_abberations(c_out)

    # We use torch.cfloat (complex64) to save VRAM.
    # If you need absolute double precision, use torch.cdouble (complex128)
    input_abberations = torch.tensor(in_abb_np, dtype=torch.cfloat, device=device)
    output_abberations = torch.tensor(out_abb_np, dtype=torch.cfloat, device=device)
    obj_t = torch.tensor(obj, dtype=torch.cfloat, device=device)

    # Output arrays natively on GPU
    k_outs = torch.zeros((N, N, N, N), dtype=torch.cfloat, device=device)
    k_ins = torch.zeros((N, N, N, N), dtype=torch.cfloat, device=device)

    # 2. Vectorized Input Grid (Batched over 'x' to save memory)
    y_idx = torch.arange(N, device=device)

    for x in range(N):
        # Create batch of impulses: Shape (N, 2N, 2N)
        k_in_batch = torch.zeros((N, 2 * N, 2 * N), dtype=torch.cfloat, device=device)
        k_in_batch[y_idx, (N // 2) + x, (N // 2) + y_idx] = 1.0

        # 3. Apply Aberrations in K-Space
        s_inc_k = k_in_batch * input_abberations

        # 4. IFFT (Transform K -> R Space)
        # Shift -> IFFT2 (over last two dims) -> Shift back
        s_inc_k_shifted = torch.fft.ifftshift(s_inc_k, dim=(-2, -1))
        s_inc_r_unshifted = torch.fft.ifft2(s_inc_k_shifted, dim=(-2, -1), norm="ortho")
        s_inc_r = torch.fft.fftshift(s_inc_r_unshifted, dim=(-2, -1))

        # 5. Object Interaction in Real Space
        s_ref_r = s_inc_r * obj_t

        # 6. FFT (Transform R -> K Space)
        s_ref_r_shifted = torch.fft.ifftshift(s_ref_r, dim=(-2, -1))
        s_ref_k_unshifted = torch.fft.fft2(s_ref_r_shifted, dim=(-2, -1), norm="ortho")
        s_ref_k = torch.fft.fftshift(s_ref_k_unshifted, dim=(-2, -1))

        # 7. Apply Output Aberrations
        s_out_k = s_ref_k * output_abberations

        # 8. Crop to N x N
        k_ins[x, :] = k_in_batch[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        k_outs[x, :] = s_out_k[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    return k_ins, k_outs
