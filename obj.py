from config import config
from skimage import data, transform, io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# usaf = io.imread("USAF-1951_65nm.png", as_gray=True)

coins = data.coins()[:303, :303]
obj = transform.resize(coins, (config.N, config.N), anti_aliasing=True)
