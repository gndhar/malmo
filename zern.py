from zernike import RZern
import numpy as np
from config import config

N: int = config.N

x = y = np.linspace(-1, 1, N)
xv, yv = np.meshgrid(x, y)


def generate_abberations(n: int, coeffs: list[float]) -> np.ndarray:
    cart = RZern(n)
    cart.make_cart_grid(xv, yv)
    c = np.zeros(cart.nk)

    coeff_count: int = min(len(coeffs), cart.nk)
    c[:coeff_count] = coeffs[:coeff_count]

    phi = cart.eval_grid(c, matrix=True)
    return np.nan_to_num(np.exp(1j * phi), nan=0.0)
