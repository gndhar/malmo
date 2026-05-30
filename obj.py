"""
obj.py  (malmo_dataset version)
--------------------------------
Provides `get_random_obj()` for per-batch random object sampling from the
pre-generated scene pool in data/scenes/.

Usage in forward_pt.py:
    from obj import get_random_obj
    obj_t = get_random_obj(device=device)   # returns torch.Tensor (2N, 2N)

Falls back to skimage coins if the scene pool hasn't been generated yet.
"""

import os
import glob
import numpy as np
import torch
from PIL import Image

from config import config

N = config.N
IMG_SIZE = 2 * N  # Forward sim works on a 2N×2N padded grid

_SCENE_DIR = os.path.join(os.path.dirname(__file__), "data", "scenes")
_scene_paths: list[str] = []

def _load_scene_pool():
    global _scene_paths
    if os.path.isdir(_SCENE_DIR):
        _scene_paths = sorted(glob.glob(os.path.join(_SCENE_DIR, "scene_*.png")))
    if not _scene_paths:
        print(
            "[obj.py] WARNING: No scene PNGs found in data/scenes/. "
            "Run generate_scenes.py first. Falling back to skimage coins."
        )

_load_scene_pool()


def _load_png(path: str) -> np.ndarray:
    """Load a 16-bit grayscale PNG and return float64 array in [0, 1]."""
    img = Image.open(path)
    arr = np.array(img, dtype=np.float64)
    arr /= 65535.0
    return arr


def _fallback_obj() -> np.ndarray:
    """Coins image resized to (2N, 2N) — used when scene pool is empty."""
    from skimage import data, transform
    base = data.coins()
    min_dim = min(base.shape)
    coins = base[:min_dim, :min_dim]
    resized = transform.resize(coins, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
    return resized.astype(np.float64)


def get_random_obj(rng: np.random.Generator | None = None,
                   device: str | torch.device = "cpu",
                   as_complex: bool = True) -> torch.Tensor:
    """
    Sample a random object image from the scene pool.

    Returns a torch.Tensor of shape (2N, 2N), dtype cfloat (if as_complex=True)
    or float32, ready to be used directly in simulate_pt_vectorized.
    """
    if _scene_paths:
        if rng is not None:
            idx = rng.integers(len(_scene_paths))
        else:
            idx = np.random.randint(len(_scene_paths))
        arr = _load_png(_scene_paths[idx])
    else:
        arr = _fallback_obj()

    dtype = torch.cfloat if as_complex else torch.float32
    return torch.tensor(arr, dtype=dtype, device=device)


def get_batch_objs(batch_size: int,
                   rng: np.random.Generator | None = None,
                   device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Sample `batch_size` random object images.

    Returns shape (batch_size, 2N, 2N), dtype cfloat.
    Each item in the batch is independently sampled.
    """
    imgs = [get_random_obj(rng=rng, device=device, as_complex=True)
            for _ in range(batch_size)]
    return torch.stack(imgs, dim=0)


# Legacy compatibility
# The original codebase imported `obj` as a plain numpy array.
# Provide it for backward compatibility with forward_sim.py etc.
# For training we use get_random_obj() / get_batch_objs() instead.
if _scene_paths:
    obj = _load_png(_scene_paths[0])   # just the first scene for static use
else:
    obj = _fallback_obj()
