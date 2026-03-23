import forward_sim
from config import config
import numpy as np
import fft
import reflection_matrix


def cass():
    s_in, s_out = forward_sim.simulate()
    N = config.N

    R, *_ = reflection_matrix.generate_R(s_in, s_out)
    R_k = reflection_matrix.RM_fft(R)

    total_k_out = np.zeros((2 * config.N, 2 * config.N), dtype=complex)

    pad_val = N // 2
    for y in range(config.N):
        for x in range(config.N):
            k_out = R_k.reshape(N, N, N, N)[:, :, y, x]
            N = config.N
            k_out = np.pad(k_out, pad_width=((pad_val, pad_val), (pad_val, pad_val)))
            x_ = (N // 2) - x
            y_ = (N // 2) - y
            k_out_shift = np.roll(k_out, shift=(y_, x_), axis=(0, 1))
            total_k_out += k_out_shift
            k_out_shift[k_out_shift == 0] = np.nan

    return fft.ifft2(total_k_out), total_k_out


if __name__ == "__main__":
    cass_r, cass_k = cass()

    import matplotlib.pyplot as plt
    import obj

    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(cass_r))
    plt.subplot(1, 2, 2)
    plt.imshow(obj.obj)
    plt.show()
