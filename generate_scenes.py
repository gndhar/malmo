"""
generate_scenes.py
------------------
Generates NUM_SCENES diverse 2D object images for training the aberration-correction network.

Each image is 2N×2N (matching forward_pt.py's padded simulation grid) containing
a random mix of shapes (Gaussian blobs, discs, rectangles, rings, point clouds)
at random z-depths. Shapes at depth z are rendered with a defocus PSF of width
proportional to |z|, giving naturalistic depth-of-field blur.

Only the final 2D intensity map is saved (no depth map stored).
Output: data/scenes/scene_{i:05d}.png   (grayscale float32, saved as 16-bit PNG)
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from config import config

#Parameters
N = config.N
IMG_SIZE = 2 * N          # 128×128 padded grid — must match forward_pt.py
NUM_SCENES = 500
DEPTH_RANGE = 8.0         # ±8 µm
SEED = 42

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "scenes")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)


#Helpers

def sigma_from_depth(z: float, sigma_min: float = 0.5, depth_scale: float = 0.3) -> float:
    """Gaussian PSF width as a function of depth (defocus model)."""
    return sigma_min + abs(z) * depth_scale


def gaussian_blob(canvas: np.ndarray, cx: float, cy: float, sigma: float,
                  intensity: float) -> None:
    """Add a 2D Gaussian blob in-place."""
    S = IMG_SIZE
    x = np.arange(S)
    y = np.arange(S)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    blob = intensity * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    canvas += blob


def draw_disc(canvas: np.ndarray, cx: float, cy: float, radius: float,
              sigma: float, intensity: float) -> None:
    """Filled disc, blurred by a Gaussian of width sigma (depth blur)."""
    S = IMG_SIZE
    x = np.arange(S)
    y = np.arange(S)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = intensity * (dist <= radius).astype(float)
    # Soft-edge blur
    from scipy.ndimage import gaussian_filter
    disc = gaussian_filter(disc, sigma=max(sigma, 0.5))
    canvas += disc


def draw_rect(canvas: np.ndarray, cx: float, cy: float, hw: float, hh: float,
              angle: float, sigma: float, intensity: float) -> None:
    """Rotated rectangle, blurred by sigma."""
    S = IMG_SIZE
    x = np.arange(S)
    y = np.arange(S)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dx = xx - cx
    dy = yy - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rx = cos_a * dx + sin_a * dy
    ry = -sin_a * dx + cos_a * dy
    mask = (np.abs(rx) <= hw) & (np.abs(ry) <= hh)
    rect = intensity * mask.astype(float)
    from scipy.ndimage import gaussian_filter
    rect = gaussian_filter(rect, sigma=max(sigma, 0.5))
    canvas += rect


def draw_ring(canvas: np.ndarray, cx: float, cy: float, r_inner: float,
              r_outer: float, sigma: float, intensity: float) -> None:
    """Annular ring, blurred by sigma."""
    S = IMG_SIZE
    x = np.arange(S)
    y = np.arange(S)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring = intensity * ((dist >= r_inner) & (dist <= r_outer)).astype(float)
    from scipy.ndimage import gaussian_filter
    ring = gaussian_filter(ring, sigma=max(sigma, 0.5))
    canvas += ring


def draw_bar(canvas: np.ndarray, cx: float, cy: float, length: float,
             width: float, angle: float, sigma: float, intensity: float) -> None:
    """Thin bar (elongated rectangle) for USAF-like patterns."""
    draw_rect(canvas, cx, cy, length / 2, width / 2, angle, sigma, intensity)


def generate_scene(rng: np.random.Generator) -> np.ndarray:
    """Generate one random 2D scene with 3–25 random shapes."""
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float64)

    n_shapes = rng.integers(3, 26)
    margin = IMG_SIZE * 0.1
    lo, hi = margin, IMG_SIZE - margin

    shape_types = ['blob', 'disc', 'rect', 'ring', 'bar', 'point_cluster']

    for _ in range(n_shapes):
        cx = rng.uniform(lo, hi)
        cy = rng.uniform(lo, hi)
        z  = rng.uniform(-DEPTH_RANGE, DEPTH_RANGE)
        sigma = sigma_from_depth(z)
        intensity = rng.uniform(0.3, 1.0)
        shape = rng.choice(shape_types)

        if shape == 'blob':
            blob_sigma = sigma + rng.uniform(0.5, 3.0)
            gaussian_blob(canvas, cx, cy, blob_sigma, intensity)

        elif shape == 'disc':
            radius = rng.uniform(2.0, IMG_SIZE * 0.12)
            draw_disc(canvas, cx, cy, radius, sigma, intensity)

        elif shape == 'rect':
            hw = rng.uniform(2.0, IMG_SIZE * 0.15)
            hh = rng.uniform(2.0, IMG_SIZE * 0.15)
            angle = rng.uniform(0, np.pi)
            draw_rect(canvas, cx, cy, hw, hh, angle, sigma, intensity)

        elif shape == 'ring':
            r_outer = rng.uniform(4.0, IMG_SIZE * 0.15)
            thickness = rng.uniform(1.0, r_outer * 0.4)
            r_inner = max(0, r_outer - thickness)
            draw_ring(canvas, cx, cy, r_inner, r_outer, sigma, intensity)

        elif shape == 'bar':
            length = rng.uniform(5.0, IMG_SIZE * 0.3)
            width  = rng.uniform(1.0, 4.0)
            angle  = rng.uniform(0, np.pi)
            draw_bar(canvas, cx, cy, length, width, angle, sigma, intensity)

        elif shape == 'point_cluster':
            # Random tight cluster of point sources
            n_pts = rng.integers(3, 12)
            spread = rng.uniform(1.0, 8.0)
            for _ in range(n_pts):
                px = cx + rng.normal(0, spread)
                py = cy + rng.normal(0, spread)
                pt_int = intensity * rng.uniform(0.4, 1.0)
                pt_sigma = sigma + rng.uniform(0.2, 1.0)
                gaussian_blob(canvas, px, py, pt_sigma, pt_int)

    # Normalize to [0, 1]
    mx = canvas.max()
    if mx > 0:
        canvas /= mx

    return canvas.astype(np.float32)


def apply_augmentation(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light augmentation applied at generation time for extra variety."""
    from scipy.ndimage import affine_transform, map_coordinates

    # Random horizontal / vertical flip
    if rng.random() < 0.5:
        img = np.fliplr(img)
    if rng.random() < 0.5:
        img = np.flipud(img)

    # Random 90° rotation
    k = rng.integers(0, 4)
    img = np.rot90(img, k=k)

    # Small random brightness tweak
    scale = rng.uniform(0.7, 1.0)
    img = (img * scale).clip(0, 1)

    return img.astype(np.float32)


def save_scene(img: np.ndarray, path: str) -> None:
    """Save float32 [0,1] array as a 16-bit grayscale PNG."""
    arr_uint16 = (img * 65535).astype(np.uint16)
    pil_img = Image.fromarray(arr_uint16, mode='I;16')
    pil_img.save(path)


#Main 

def main():
    print(f"Generating {NUM_SCENES} scenes of size {IMG_SIZE}×{IMG_SIZE} "
          f"into {OUT_DIR} ...")

    for i in tqdm(range(NUM_SCENES)):
        scene = generate_scene(rng)
        scene = apply_augmentation(scene, rng)
        out_path = os.path.join(OUT_DIR, f"scene_{i:05d}.png")
        save_scene(scene, out_path)

    print(f"\nDone. {NUM_SCENES} scenes saved to {OUT_DIR}")
    print(f"Approx. disk usage: {NUM_SCENES * IMG_SIZE * IMG_SIZE * 2 / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
