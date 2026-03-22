import numpy as np
import fft
from obj import obj
from zern import generate_abberations
import zern
import enum
from config import config

c_in = list(np.random.random(zern.cart.nk))
c_out = list(np.random.random(zern.cart.nk))
# c_in = list(np.zeros((zern.cart.nk), dtype=float))
# c_out = list(np.zeros((zern.cart.nk), dtype=float))

input_abberations = generate_abberations(c_in)
output_abberations = generate_abberations(c_out)


class Space(enum.Enum):
    K = enum.auto()
    R = enum.auto()


class Signal:
    def __init__(self, data: np.ndarray, space: Space):
        self.space = space
        if space == Space.R:
            self.r = data
            self.k = fft.fft2(data)
        elif space == Space.K:
            self.k = data
            self.r = fft.ifft2(data)


def simulate() -> tuple[Signal, Signal]:
    global input_abberations, output_abberations

    N = config.N

    k_outs = np.zeros((N, N, N, N), dtype=complex)
    k_ins = np.zeros((N, N, N, N), dtype=complex)

    for x in range(N):
        for y in range(N):
            k_in_padded = np.zeros((N * 2, N * 2))
            k_in_padded[N // 2 + x, N // 2 + y] = 1.0

            # store inputs
            k_ins[x, y] = k_in_padded[N // 2 : N // 2 + N, N // 2 : N // 2 + N]

            s_in = Signal(k_in_padded, Space.K)
            s_inc = Signal(s_in.k * input_abberations, Space.K)
            s_ref = Signal(s_inc.r * obj, Space.R)
            s_out = Signal(s_ref.k * output_abberations, Space.K)

            # store outputs
            k_outs[x, y] = s_out.k[N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    return Signal(k_ins, Space.K), Signal(k_outs, Space.K)


if __name__ == "__main__":
    ...
