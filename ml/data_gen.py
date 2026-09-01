"""Data generation for MALMO: synthetic myelin-fiber objects + complex phase
aberration maps.

Key Features & Updates:
- `generate_high_depth_grf`: Generates continuous smooth GRF phase fields.
- `HighDepthGRFGenerator`: Produces zero-mean GRF complex phase maps masked inside the active pupil.
- `RMDataset.__getitem__`: Directly yields (ab_in, ab_out, obj) complex phase fields of shape (N, N).
- Monitoring shape print statement in `RMDataset.__getitem__`.
"""

import math
import multiprocessing as mp
import os
import random
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from zern import ZernikeAberration


def generate_high_depth_grf(
    batch_size: int,
    N: int = 64,
    kernel_size: int = 17,
    sigma_spatial: float = 2.0,
    min_wraps: float = 2.0,
    max_wraps: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates continuous smooth phase fields on an N x N grid with 2pi wrap controls."""
    coords = (
        torch.arange(kernel_size, dtype=torch.float32, device=device)
        - (kernel_size - 1) / 2
    )
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    gaussian_kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_spatial**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    noise = torch.randn(batch_size, 1, N, N, device=device)
    smoothed = F.conv2d(noise, gaussian_kernel, padding=kernel_size // 2)

    smoothed_std = smoothed.std(dim=(-2, -1), keepdim=True) + 1e-8
    smooth_normalized = smoothed / smoothed_std

    target_wraps = (
        torch.empty(batch_size, 1, 1, 1, device=device).uniform_(min_wraps, max_wraps)
        * 2
        * torch.pi
    )
    ptv_current = smooth_normalized.amax(
        dim=(-2, -1), keepdim=True
    ) - smooth_normalized.amin(dim=(-2, -1), keepdim=True)

    continuous_phase = (smooth_normalized / ptv_current) * target_wraps
    return continuous_phase.squeeze(1)


class HighDepthGRFGenerator:
    """Generator for creating GRF complex phase maps with zero mean phase inside the pupil mask."""

    def __init__(
        self,
        N: int = 64,
        kernel_size: int = 17,
        sigma_spatial: float = 2.0,
        min_wraps: float = 2.0,
        max_wraps: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.N = N
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.min_wraps = min_wraps
        self.max_wraps = max_wraps
        self.device = device

        # Pupil mask from ZernikeAberration (N // 2 creates an N x N grid mask)
        self.zern_gen = ZernikeAberration(N=N // 2, zern_n=0)
        self.pupil_mask = self.zern_gen.pupil_mask.to(device)
        self.active_mask = self.pupil_mask > 0

    def __call__(self) -> torch.Tensor:
        phase = generate_high_depth_grf(
            batch_size=1,
            N=self.N,
            kernel_size=self.kernel_size,
            sigma_spatial=self.sigma_spatial,
            min_wraps=self.min_wraps,
            max_wraps=self.max_wraps,
            device=self.device,
        )[0]

        pupil_mean = phase[self.active_mask].mean()
        zero_mean_phase = phase - pupil_mean

        complex_phase = torch.polar(torch.ones_like(zero_mean_phase), zero_mean_phase)
        return torch.where(
            self.active_mask,
            complex_phase,
            torch.tensor(0.0 + 0.0j, device=self.device),
        )


class StandardZernikeGenerator:
    """Default generator for creating Zernike complex phase maps."""

    def __init__(self, N: int, zern_n: int):
        self.ab_gen = ZernikeAberration(N // 2, zern_n=zern_n)
        self.coeff_count = self.ab_gen.num_coefficients

    def __call__(self) -> torch.Tensor:
        coeffs = torch.rand(self.coeff_count) * 2 - 1
        coeffs[0] = 0.0  # Zero out piston term
        return self.ab_gen(coeffs)


def _rescale(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin) < 1e-12:
        return torch.full_like(x, lo)
    return (x - xmin) / (xmax - xmin) * (hi - lo) + lo


def create_myelin_target(
    N: int,
    num_fibers: int,
    rng: random.Random,
    device: str = "cpu",
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Synthetic myelin-fiber-like complex reflectivity target on an N x N grid."""
    thickness = 0.7
    min_len = 0.25 * N
    min_fiber_amp = 0.4
    phase_choices = (-1.0, -0.5, 0.5, 1.0)

    chunk_size = min(chunk_size, max(256, int(1.25e7 / (N * N))))

    endpoints = []
    for _ in range(num_fibers):
        while True:
            x0, y0, x1, y1 = (rng.random() * N for _ in range(4))
            if math.hypot(x1 - x0, y1 - y0) >= min_len:
                break
        endpoints.append((x0, y0, x1, y1))

    xs, ys, amps, phases = [], [], [], []

    for x, y, x_end, y_end in endpoints:
        fiber_len = int(round(math.hypot(x_end - x, y_end - y)))
        if fiber_len < 1:
            continue

        angle = math.atan2(y_end - y, x_end - x)
        angle_deltas = [rng.gauss(0.0, 0.1) for _ in range(fiber_len)]
        fiber_amps = [rng.uniform(min_fiber_amp, 1.0) for _ in range(fiber_len)]
        phase_picks = [rng.choice(phase_choices) for _ in range(fiber_len)]
        taper = [
            math.exp(-(((s + 1) - fiber_len / 2) ** 2) / (0.6 * fiber_len) ** 2)
            for s in range(fiber_len)
        ]

        for step in range(fiber_len):
            angle += angle_deltas[step]
            x += math.cos(angle)
            y += math.sin(angle)
            if x < 0 or x > N - 1 or y < 0 or y > N - 1:
                break

            xs.append(x)
            ys.append(y)
            amps.append(fiber_amps[step] * taper[step])
            phases.append(math.pi * phase_picks[step])

    if not xs:
        z = torch.zeros(N, N, device=device)
        return torch.polar(z, z)

    pts_x = torch.tensor(xs, device=device, dtype=torch.float32)
    pts_y = torch.tensor(ys, device=device, dtype=torch.float32)
    pts_a = torch.tensor(amps, device=device, dtype=torch.float32)
    pts_p = torch.tensor(phases, device=device, dtype=torch.float32)

    coords = torch.arange(N, device=device, dtype=torch.float32)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")

    amp_map = torch.zeros(N, N, device=device)
    phase_map = torch.zeros(N, N, device=device)

    num_points = pts_x.shape[0]
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dx = X.unsqueeze(0) - pts_x[start:end].view(-1, 1, 1)
        dy = Y.unsqueeze(0) - pts_y[start:end].view(-1, 1, 1)
        blobs = torch.exp(-(dx**2 + dy**2) / (2 * thickness**2))
        amp_map += (blobs * pts_a[start:end].view(-1, 1, 1)).sum(dim=0)
        phase_map += (blobs * pts_p[start:end].view(-1, 1, 1)).sum(dim=0)

    amp_map = _rescale(amp_map, 0.0, 1.0)
    phase_map = _rescale(phase_map, -math.pi, math.pi)

    return torch.polar(amp_map, phase_map)


class ObjDataset(Dataset):
    """On-the-fly (uncached) object generation."""

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

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = random.Random(self.seed + idx)
        num_fibers = rng.randint(self.min_fibers, self.max_fibers)
        return create_myelin_target(self.N, num_fibers, rng)


def _generate_one(args):
    """Module-level worker for multiprocessing.Pool object cache builds."""
    idx, N, seed, min_fibers, max_fibers = args
    rng = random.Random(seed + idx)
    num_fibers = rng.randint(min_fibers, max_fibers)
    return idx, create_myelin_target(N, num_fibers, rng)


class RMDataset(Dataset):
    """Dataset producing input phase maps, output phase maps, and synthetic objects.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (ab_in, ab_out, obj)
        all as complex tensors of shape (N, N).
    """

    def __init__(
        self,
        N: int,
        size: int,
        zern_n: int = 5,
        aberration_type: str = "grf",
        seed: int = 42,
        min_fibers: int = 5,
        max_fibers: int = 20,
        cache_objects: bool = True,
        cache_path: Optional[str] = None,
        num_workers_build: int = 1,
        aberration_generator: Optional[Callable[[], torch.Tensor]] = None,
        grf_kernel_size: int = 17,
        grf_sigma_spatial: float = 2.0,
        grf_min_wraps: float = 2.0,
        grf_max_wraps: float = 4.0,
    ):
        self.N = N
        self.size = size
        self.seed = seed
        self.cache_objects = cache_objects

        # Setup Object Source
        if cache_objects:
            self._cached_objs = self._build_or_load_cache(
                N,
                size,
                seed,
                min_fibers,
                max_fibers,
                cache_path,
                num_workers_build,
            )
        else:
            self.obj_dataset = ObjDataset(N, size, seed, min_fibers, max_fibers)

        # Setup Aberration Generator Strategy
        if aberration_generator is not None:
            self.aberration_generator = aberration_generator
        elif aberration_type == "grf":
            self.aberration_generator = HighDepthGRFGenerator(
                N=N,
                kernel_size=grf_kernel_size,
                sigma_spatial=grf_sigma_spatial,
                min_wraps=grf_min_wraps,
                max_wraps=grf_max_wraps,
            )
        elif aberration_type == "zernike":
            self.aberration_generator = StandardZernikeGenerator(N=N, zern_n=zern_n)
        else:
            raise ValueError(f"Unknown aberration_type: {aberration_type!r}")

    @staticmethod
    def _build_or_load_cache(
        N: int,
        size: int,
        seed: int,
        min_fibers: int,
        max_fibers: int,
        cache_path: Optional[str],
        num_workers_build: int,
    ) -> torch.Tensor:
        if cache_path is not None and os.path.exists(cache_path):
            print(f"Loading cached objects from {cache_path}")
            return torch.load(cache_path)

        if num_workers_build > 1:
            args = [(i, N, seed, min_fibers, max_fibers) for i in range(size)]
            objs_by_idx = {}
            ctx = mp.get_context("spawn")
            with ctx.Pool(num_workers_build) as pool:
                for idx, obj in tqdm(
                    pool.imap_unordered(_generate_one, args, chunksize=8),
                    total=size,
                    desc="Caching objects",
                ):
                    objs_by_idx[idx] = obj
            objs = [objs_by_idx[i] for i in range(size)]
        else:
            objs = []
            for idx in tqdm(range(size), desc="Caching objects"):
                rng = random.Random(seed + idx)
                num_fibers = rng.randint(min_fibers, max_fibers)
                objs.append(create_myelin_target(N, num_fibers, rng))

        cached = torch.stack(objs)

        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(cached, cache_path)
            print(f"Saved cached objects to {cache_path}")

        return cached

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obj = self._cached_objs[idx] if self.cache_objects else self.obj_dataset[idx]

        ab_in = self.aberration_generator()
        ab_out = self.aberration_generator()

        return ab_in, ab_out, obj
