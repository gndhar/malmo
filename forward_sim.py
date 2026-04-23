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


def simulate_optimized(c_in=c_in, c_out=c_out) -> tuple[Signal, Signal]:
    N = config.N

    # 1. Generate aberrations once for the whole batch
    input_abberations = generate_abberations(c_in)
    output_abberations = generate_abberations(c_out)

    # 2. Vectorized Input Grid (The "Delta" impulses)
    # Shape: (N, N, 2N, 2N)
    k_in_batch = np.zeros((N, N, 2 * N, 2 * N), dtype=complex)

    # Broadcast the indices to place the 1.0s without a loop
    x_idx = np.arange(N)[:, None]  # Column vector for broadcasting (N, 1)
    y_idx = np.arange(N)  # Row vector for broadcasting (N,)
    k_in_batch[x_idx, y_idx, (N // 2) + x_idx, (N // 2) + y_idx] = 1.0

    # 3. Apply input aberrations in K-space
    # Broadcasting handles (N, N, 2N, 2N) * (2N, 2N)
    s_inc_k = k_in_batch * input_abberations

    # 4. Transform to Real Space (Batch IFFT)
    # Your custom fft.ifft2 uses axes=(-2, -1), so it applies the 2D IFFT
    # to all N*N matrices simultaneously.
    s_inc_r = fft.ifft2(s_inc_k)

    # 5. Interact with Object in Real Space
    s_ref_r = s_inc_r * obj

    # 6. Transform back to K-space (Batch FFT)
    s_ref_k = fft.fft2(s_ref_r)

    # 7. Apply output aberrations
    s_out_k = s_ref_k * output_abberations

    # 8. Crop to N x N
    k_ins = k_in_batch[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
    k_outs = s_out_k[:, :, N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    # Return as Signals.
    # Because your Signal class also relies on fft.ifft2, passing the 4D
    # cropped arrays will correctly generate the 4D real-space counterparts.
    return Signal(k_ins, Space.K), Signal(k_outs, Space.K)


def generate_visual_simulation():
    global input_abberations, output_abberations

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,  # Set to True only if you have a full LaTeX distro (MikTeX/TeXLive) installed
            "mathtext.fontset": "cm",  # 'cm' stands for Computer Modern
            "font.family": "serif",
            "axes.formatter.use_mathtext": True,
        }
    )

    N = config.N

    k_outs = np.zeros((N, N, N, N), dtype=complex)
    k_ins = np.zeros((N, N, N, N), dtype=complex)

    fig, axes = plt.subplots(3, 5)
    frames = []

    # --- Column 0: Initial State (Pure Plane Wave) ---
    axes[0, 0].set_title(r"$|\tilde{E}_{in}(\mathbf{k})|$")
    axes[1, 0].set_title(r"$|E_{in}(\mathbf{r})|$")
    axes[2, 0].set_title(r"$\angle E_{in}(\mathbf{r})$")

    # --- Column 1: Incident Wave (After Input Aberrations) ---
    axes[0, 1].set_title(r"$|\tilde{E}_{inc}(\mathbf{k})|$")
    axes[1, 1].set_title(r"$|E_{inc}(\mathbf{r})|$")
    axes[2, 1].set_title(r"$\angle E_{inc}(\mathbf{r})$")

    # --- Column 2: Reflected Wave (Object Interaction) ---
    axes[0, 2].set_title(r"$|\tilde{E}_{ref}(\mathbf{k})|$")
    axes[1, 2].set_title(r"$|E_{ref}(\mathbf{r})|$")
    axes[2, 2].set_title(r"$\angle E_{ref}(\mathbf{r})$")

    # --- Column 3: Output & Detection (After Output Aberrations + Crop) ---
    axes[0, 3].set_title(r"$|\tilde{E}_{out}(\mathbf{k})|$")
    axes[1, 3].set_title(r"$|E_{out}(\mathbf{r})|$")
    axes[2, 3].set_title(
        r"$|E_{out,clipped}(\mathbf{r})|$"
    )  # Detected field (reconstructed from crop)

    # --- Column 4: System Parameters (Ground Truths) ---
    axes[0, 4].set_title(r"$\angle \theta_{in}(\mathbf{k})$")  # Input Phase Aberration
    axes[1, 4].set_title(
        r"$\angle \theta_{out}(\mathbf{k})$"
    )  # Output Phase Aberration
    axes[2, 4].set_title(r"$O(\mathbf{r})$")  # Ground Truth Object

    for x in range(N):
        y = x
        ims = []
        k_in_padded = np.zeros((N * 2, N * 2))
        k_in_padded[N // 2 + x, N // 2 + y] = 1.0

        # store inputs
        k_ins[x, y] = k_in_padded[N // 2 : N // 2 + N, N // 2 : N // 2 + N]

        s_in = Signal(k_in_padded, Space.K)
        ims.append(axes[0, 0].imshow(np.abs(s_in.k), vmin=0))
        ims.append(axes[1, 0].imshow(np.abs(s_in.r), vmin=0))
        ims.append(
            axes[2, 0].imshow(
                np.angle(s_in.r), vmin=-np.pi, vmax=np.pi, cmap="twilight"
            )
        )
        s_inc = Signal(s_in.k * input_abberations, Space.K)
        ims.append(axes[0, 1].imshow(np.abs(s_inc.k), vmin=0))
        ims.append(axes[1, 1].imshow(np.abs(s_inc.r), vmin=0))
        ims.append(
            axes[2, 1].imshow(
                np.angle(s_inc.r), vmin=-np.pi, vmax=np.pi, cmap="twilight"
            )
        )
        s_ref = Signal(s_inc.r * obj, Space.R)
        ims.append(axes[0, 2].imshow(np.abs(s_ref.k), vmin=0))
        ims.append(axes[1, 2].imshow(np.abs(s_ref.r), vmin=0))
        ims.append(
            axes[2, 2].imshow(
                np.angle(s_ref.r), vmin=-np.pi, vmax=np.pi, cmap="twilight"
            )
        )
        s_out = Signal(s_ref.k * output_abberations, Space.K)

        # store outputs
        k_outs[x, y] = s_out.k[N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        ims.append(axes[0, -2].imshow(np.abs(s_out.k), vmin=0))
        ims.append(axes[1, -2].imshow(np.abs(s_out.r), vmin=0))
        ims.append(axes[2, -2].imshow(np.abs(fft.ifft2(k_outs[x, y]))))
        ims.append(axes[0, -1].imshow(np.angle(input_abberations), cmap="twilight"))
        ims.append(axes[1, -1].imshow(np.angle(output_abberations), cmap="twilight"))
        ims.append(axes[2, -1].imshow(obj, vmin=0))
        frames.append(ims)

    from matplotlib.animation import ArtistAnimation

    anim = ArtistAnimation(fig, frames)
    plt.show()

    # return Signal(k_ins, Space.K), Signal(k_outs, Space.K)


def simulate_batched(c_in=c_in, c_out=c_out):
    N = config.N

    # Generate aberrations once
    input_abberations = generate_abberations(c_in)
    output_abberations = generate_abberations(c_out)

    k_outs = np.zeros((N, N, N, N), dtype=complex)
    k_ins = np.zeros((N, N, N, N), dtype=complex)

    # Mini-batch over the 'x' dimension.
    # This reduces loop overhead by 64x without blowing up RAM.
    for x in range(N):
        # 1. Create a batch for just ONE row: Shape (N, 2N, 2N)
        k_in_batch = np.zeros((N, 2 * N, 2 * N), dtype=complex)

        # Place the delta impulses across the 'y' dimension
        y_idx = np.arange(N)
        k_in_batch[y_idx, (N // 2) + x, (N // 2) + y_idx] = 1.0

        # 2. Vectorized Forward Pass (operating on N arrays simultaneously)
        s_inc_k = k_in_batch * input_abberations
        s_inc_r = fft.ifft2(s_inc_k)
        s_ref_r = s_inc_r * obj
        s_ref_k = fft.fft2(s_ref_r)
        s_out_k = s_ref_k * output_abberations

        # 3. Crop and store directly into the output arrays
        k_ins[x, :] = k_in_batch[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        k_outs[x, :] = s_out_k[:, N // 2 : N // 2 + N, N // 2 : N // 2 + N]

    # Return final N x N x N x N signals
    return Signal(k_ins, Space.K), Signal(k_outs, Space.K)


if __name__ == "__main__":
    # generate_visual_simulation()
    import time

    t0 = time.time()
    for _ in range(10):
        simulate()
    print("done 1")
    t1 = time.time()
    for _ in range(10):
        simulate_optimized()
    print("done 2")
    t2 = time.time()
    for _ in range(10):
        simulate_batched()
    print("done 3")
    print(f"{t1 - t0} {t2 - t1} {time.time() - t2}")
