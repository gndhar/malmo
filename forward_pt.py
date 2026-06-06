import torch
import torch.fft
import numpy as np

from config import config
from zern import generate_abberations
from obj import obj
from forward_sim import c_in, c_out


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


if __name__ == "__main__":
    import time

    start = time.time()
    device = "mps"
    N = 10
    for _ in range(N):
        simulate_pt(device=device)
    t1 = time.time()
    print((t1 - start) / N)
