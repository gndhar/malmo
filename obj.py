from config import config
import numpy as np
from skimage import data, transform, io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
N = config.N

# usaf = io.imread("USAF-1951_65nm.png", as_gray=True)

base_img = data.coins()
min_dim = min(base_img.shape)
coins = base_img[:min_dim, :min_dim]
obj = transform.resize(coins, (2 * config.N, 2 * config.N), anti_aliasing=True).astype(
    complex
)
# obj *= np.exp(1j * np.random.uniform(0, 2 * np.pi, (2 * N, 2 * N)))
