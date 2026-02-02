import numpy as np
from fft import *
from config import config
from obj import obj
from zern import generate_abberations

N = config.N
r_in = np.zeros((N, N, N, N), dtype=np.float64)

idx = np.arange(N)
r_in[idx[:, None], idx, idx[:, None], idx] = 1.0


# clipping in k space
x, y = np.ogrid[:N, :N]
r2 = (x - (N // 2)) ** 2 + (y - (N // 2)) ** 2
clipping_mask = r2 <= (N // 2) ** 2

c = [0.0] * 13

c[3] = 0.50
c[4] = 0.15
c[5] = 0.12
c[6] = 0.18
c[7] = 0.14
c[10] = 0.11
c[12] = 0.08


def propagate_r_r(r_in: np.ndarray) -> np.ndarray:
    r_incident = input_path(r_in)
    r_reflected = r_incident * obj
    return output_path(r_reflected)


def propagate_k_r(k_in: np.ndarray) -> np.ndarray:
    r_incident = input_path_k(k_in)
    r_reflected = r_incident * obj
    return output_path(r_reflected)


def input_path(r_in: np.ndarray) -> np.ndarray:
    input_abberations = generate_abberations(6, c)
    k_in = fft2(r_in)
    k_in_clipped = k_in * input_abberations
    return ifft2(k_in_clipped)  # airy disk


def input_path_k(k_in: np.ndarray) -> np.ndarray:
    input_abberations = generate_abberations(6, c)
    k_in_clipped = k_in * input_abberations
    return ifft2(k_in_clipped)  # airy disk


def output_path(r: np.ndarray) -> np.ndarray:  # r = airy_disk * obj
    output_abberations = generate_abberations(6, c)
    k = fft2(r)
    k_clipped = k * output_abberations
    return ifft2(k_clipped)  # r_out


if __name__ == "__main__":
    # r_out = propagate_r_r(r_in)

    k_in = np.zeros((config.N, config.N))
    k_in[31, 12] = 1.0

    r_out = propagate_k_r(k_in)

    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(k_in)
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(r_out))
    plt.show()
