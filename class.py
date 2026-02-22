import numpy as np
import forward_sim
import fft
from config import config


def matrix_encode(k_out: np.ndarray) -> np.ndarray:
    temp = k_out.reshape(*k_out.shape[:-2], -1)
    M_kin_kout = temp.reshape(-1, *temp.shape[2:])
    return M_kin_kout


def matrix_decode(M: np.ndarray) -> np.ndarray:
    temp_recovered = M.reshape(config.N, config.N, -1)
    k_out_recovered = temp_recovered.reshape(config.N, config.N, config.N, config.N)
    return k_out_recovered


def class_solver(k_out: np.ndarray, itN: int, pow_itN):
    M_matrix = matrix_encode(k_out)

    ain_total = np.ones((config.N, config.N), dtype=complex)
    aout_total = np.ones((config.N, config.N), dtype=complex)

    for _ in range(itN):
        ph_in = power_iteration(M_matrix.T, pow_itN, ain_total.flatten())
        abi = np.exp(1j * np.angle(ph_in)).reshape(config.N, config.N)

        ain_total *= abi

        M_matrix = M_matrix * abi.flatten()

        M_conjugated = M_matrix.conj().T

        ph_out = power_iteration(M_conjugated.T, pow_itN, aout_total.flatten())
        abo = np.exp(1j * np.angle(ph_out)).reshape(config.N, config.N)

        aout_total *= abo
        M_matrix = abo.flatten()[:, np.newaxis] * M_matrix

    k_out_corrected = matrix_decode(M_matrix)
    return k_out_corrected, ain_total, aout_total


def power_iteration(T: np.ndarray, n: np.ndarray, b_init: np.ndarray) -> np.ndarray:
    b = b_init / np.linalg.norm(b_init)

    for _ in range(n):
        b_next = T @ b
        b = b_next / np.linalg.norm(b_next)

    return b


def get_r_out_cass(k_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(k_out.shape)
    k_out_shifted = np.zeros_like(k_out)
    pos = ((i, j) for i in range(config.N) for j in range(config.N))
    for x, y in pos:
        k_out_shifted[x, y] = np.roll(k_out[x, y], shift=(-x, -y), axis=(0, 1))

    import matplotlib.pyplot as plt

    plt.imshow(np.abs(k_out_shifted[31, 12]))
    plt.show()
    final_k_out = np.sum(k_out_shifted, axis=(0, 1))
    final_r_out = fft.ifft2(final_k_out)

    return final_r_out, final_k_out


if __name__ == "__main__":
    k_in = np.zeros((config.N, config.N, config.N, config.N))
    for x in range(config.N):
        for y in range(config.N):
            k_in[x, y] = 1.0

    apmask = np.where(np.abs(forward_sim.input_abberations) > 0.5)
    k_out = forward_sim.propagate_k_k(k_in)

    k_out_corrected, ain, aout = class_solver(k_out, 5, 5)
    r_out_final, _ = get_r_out_cass(k_out_corrected)

    import matplotlib.pyplot as plt

    plt.subplot(1, 2, 1)
    plt.imshow(np.angle(ain))
    plt.subplot(1, 2, 2)
    plt.imshow(np.angle(forward_sim.input_abberations))
    plt.show()
