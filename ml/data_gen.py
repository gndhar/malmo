"""Data generation for MALMO: myelin-fiber synthetic objects + reflection-matrix
Zernike coefficient pairs.
"""

import math

import torch
from torch.utils.data import Dataset

from zern import ZernikeAberration


def _rescale(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin) < 1e-12:
        return torch.full_like(x, lo)
    return (x - xmin) / (xmax - xmin) * (hi - lo) + lo


def create_myelin_target(
    N: int, num_fibers: int, generator: torch.Generator, device="cpu"
) -> torch.Tensor:
    """Synthetic myelin-fiber-like complex reflectivity target on an N x N grid.

    Ported from PC_Create_Myelin_Target.m. Point recurrence (position/angle)
    is inherently sequential and stays a Python loop over scalars (cheap).
    Blob synthesis is vectorized across all accumulated points in one
    broadcast, instead of one full (N,N) add per growth step.
    """
    thickness = 0.7
    min_len = 0.25 * N
    min_fiber_amp = 0.4

    endpoints = torch.zeros(num_fibers, 4, device=device)
    for k in range(num_fibers):
        while True:
            pts = torch.rand(4, generator=generator, device=device) * N
            if torch.hypot(pts[2] - pts[0], pts[3] - pts[1]) >= min_len:
                break
        endpoints[k] = pts

    xs, ys, amps, phases = [], [], [], []
    phase_choices = torch.tensor([-1.0, -0.5, 0.5, 1.0], device=device)

    for k in range(num_fibers):
        x, y = endpoints[k, 0].item(), endpoints[k, 1].item()
        x_end, y_end = endpoints[k, 2].item(), endpoints[k, 3].item()
        fiber_len = int(round(math.hypot(x_end - x, y_end - y)))
        if fiber_len < 1:
            continue

        angle = math.atan2(y_end - y, x_end - x)
        taper = [
            math.exp(-(((s + 1) - fiber_len / 2) ** 2) / (0.6 * fiber_len) ** 2)
            for s in range(fiber_len)
        ]

        for step in range(fiber_len):
            angle += 0.1 * torch.randn(1, generator=generator, device=device).item()
            x += math.cos(angle)
            y += math.sin(angle)
            if x < 0 or x > N - 1 or y < 0 or y > N - 1:
                break

            fiber_amp = (
                torch.rand(1, generator=generator, device=device).item()
                * (1.0 - min_fiber_amp)
                + min_fiber_amp
            )
            a = fiber_amp * taper[step]
            p = (
                math.pi
                * phase_choices[
                    torch.randint(0, 4, (1,), generator=generator, device=device)
                ].item()
            )

            xs.append(x)
            ys.append(y)
            amps.append(a)
            phases.append(p)

    if not xs:  # degenerate case: no fiber survived even one step
        z = torch.zeros(N, N, device=device)
        return torch.polar(z, z)

    pts_x = torch.tensor(xs, device=device)
    pts_y = torch.tensor(ys, device=device)
    pts_a = torch.tensor(amps, device=device)
    pts_p = torch.tensor(phases, device=device)

    coords = torch.arange(N, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")

    # (P, N, N) via broadcast -- vectorized blob batch instead of a Python loop.
    # NOTE: for large N * num_points this can get memory-heavy; chunk over
    # pts_x/pts_y in groups if you hit that.
    dx = X.unsqueeze(0) - pts_x.view(-1, 1, 1)
    dy = Y.unsqueeze(0) - pts_y.view(-1, 1, 1)
    blobs = torch.exp(-(dx**2 + dy**2) / (2 * thickness**2))

    amp_map = (blobs * pts_a.view(-1, 1, 1)).sum(dim=0)
    phase_map = (blobs * pts_p.view(-1, 1, 1)).sum(dim=0)

    amp_map = _rescale(amp_map, 0.0, 1.0)
    phase_map = _rescale(phase_map, -math.pi, math.pi)

    return torch.polar(amp_map, phase_map)  # complex = amp * exp(i*phase)


class ObjDataset(Dataset):
    def __init__(
        self,
        N: int,
        size: int,
        seed: int = 42,
        min_fibers: int = 5,
        max_fibers: int = 20,
    ):
        self.N = N
        self.size = size
        self.seed = seed
        self.min_fibers = min_fibers
        self.max_fibers = max_fibers

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        generator = torch.Generator().manual_seed(self.seed + idx)
        num_fibers = torch.randint(
            self.min_fibers, self.max_fibers + 1, (1,), generator=generator
        ).item()
        return create_myelin_target(self.N, num_fibers, generator)


class RMDataset(Dataset):
    def __init__(self, N: int, size: int, zern_n: int, seed: int = 42):
        self.N = N
        self.size = size
        self.seed = seed
        self.obj_dataset = ObjDataset(N, size, seed)

        # Only needed here for coefficient count -- Simulation itself belongs
        # in the training loop, not the dataset (it was constructed but never
        # used here before).
        ab_gen = ZernikeAberration(N, zern_n=zern_n)
        self.coeff_count = ab_gen.num_coefficients

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        obj = self.obj_dataset[idx]

        # TODO (parked, not forgotten): uniform-per-coefficient sampling.
        # Revisit radial-order-dependent (Kolmogorov/Noll) magnitude decay
        # once the rest of the pipeline is validated end-to-end.
        c_in = torch.rand(self.coeff_count) * 2 - 1
        c_out = torch.rand(self.coeff_count) * 2 - 1
        c_in[0] = 0.0
        c_out[0] = 0.0

        return c_in, c_out, obj
