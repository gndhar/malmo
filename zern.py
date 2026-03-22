from zernike import RZern
import numpy as np
from config import config

x = y = np.linspace(-2.0, 2.0, 2 * config.N)
xv, yv = np.meshgrid(x, y)

cart = RZern(config.zern_n)
cart.make_cart_grid(xv, yv)


def generate_abberations(coeffs: list[float]) -> np.ndarray:
    c = np.zeros(cart.nk)

    coeff_count: int = min(len(coeffs), cart.nk)
    c[:coeff_count] = coeffs[:coeff_count]

    phi = cart.eval_grid(c, matrix=True)
    return np.nan_to_num(np.exp(1j * phi), nan=0.0)
