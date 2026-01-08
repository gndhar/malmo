from config import config
from skimage import data, transform

N = config.N

coins = data.coins()[:303, :303]
obj = transform.resize(coins, (N, N), anti_aliasing=True)
