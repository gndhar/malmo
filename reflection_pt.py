import torch
from config import config


def generate_R_k_pt(k_ins: torch.Tensor, k_outs: torch.Tensor):
    B = k_outs.shape[0]
    N = config.N

    # 1. Reshape creates zero-copy memory views on the GPU
    # V_A is shape (1, N^2, N^2)
    # V_B is shape (B, N^2, N^2)
    V_A = k_ins.reshape(1, N * N, N * N)
    V_B = k_outs.reshape(B, N * N, N * N)

    # 2. Batched matrix multiplication.
    # Because V_A has a batch size of 1, PyTorch's @ operator automatically
    # broadcasts it against all B items in V_B without consuming extra VRAM!
    R_k = V_B.mT @ V_A.conj()

    return R_k
