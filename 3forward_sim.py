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


import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def generate_visual_simulation():
    global input_abberations, output_abberations, obj

    # 1. LaTeX Font & Aesthetic Setup
    plt.rcParams.update(
        {
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 11,  # Slightly smaller titles to prevent overlap
        }
    )

    N = config.N
    # Increased figsize width to 26 to give the 5 columns breathing room
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # Increase horizontal (wspace) and vertical (hspace) spacing explicitly
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    frames = []

    # --- Static LaTeX Titles ---
    titles = [
        [
            r"$|\tilde{E}_{in}(\mathbf{k})|$",
            r"$|\tilde{E}_{inc}(\mathbf{k})|$",
            r"$|\tilde{E}_{ref}(\mathbf{k})|$",
            r"$|\tilde{E}_{out}(\mathbf{k})|$",
            r"$\angle \theta_{in}(\mathbf{k})$",
        ],
        [
            r"$|E_{in}(\mathbf{r})|$",
            r"$|E_{inc}(\mathbf{r})|$",
            r"$|E_{ref}(\mathbf{r})|$",
            r"$|E_{out}(\mathbf{r})|$",
            r"$\angle \theta_{out}(\mathbf{k})$",
        ],
        [
            r"$\angle E_{in}(\mathbf{r})$",
            r"$\angle E_{inc}(\mathbf{r})$",
            r"$\angle E_{ref}(\mathbf{r})$",
            r"$|E_{out,clipped}(\mathbf{r})|$",
            r"$O(\mathbf{r})$",
        ],
    ]

    cbar_initialized = False

    for x in range(N - 1):
        x = (x + N // 2) % N
        y = x
        ims = []
        k_in_padded = np.zeros((N * 2, N * 2), dtype=complex)
        k_in_padded[N // 2 + x, N // 2 + y] = 1.0

        # --- Forward Simulation Pipeline ---
        s_in = Signal(k_in_padded, Space.K)
        s_inc = Signal(s_in.k * input_abberations, Space.K)
        s_ref = Signal(s_inc.r * obj, Space.R)
        s_out = Signal(s_ref.k * output_abberations, Space.K)

        k_out_clipped = s_out.k[N // 2 : N // 2 + N, N // 2 : N // 2 + N]
        e_det_mag = np.abs(fft.ifft2(k_out_clipped))

        # --- Plot Mapping ---
        plot_steps = [
            (np.abs(s_in.k), axes[0, 0], "viridis", 0, 1),
            (np.abs(s_inc.k), axes[0, 1], "viridis", 0, 1),
            (np.abs(s_ref.k), axes[0, 2], "viridis", 0, None),
            (np.abs(s_out.k), axes[0, 3], "viridis", 0, None),
            (np.angle(input_abberations), axes[0, 4], "twilight", -np.pi, np.pi),
            (np.abs(s_in.r), axes[1, 0], "viridis", 0, None),
            (np.abs(s_inc.r), axes[1, 1], "viridis", 0, None),
            (np.abs(s_ref.r), axes[1, 2], "viridis", 0, None),
            (np.abs(s_out.r), axes[1, 3], "viridis", 0, None),
            (np.angle(output_abberations), axes[1, 4], "twilight", -np.pi, np.pi),
            (np.angle(s_in.r), axes[2, 0], "twilight", -np.pi, np.pi),
            (np.angle(s_inc.r), axes[2, 1], "twilight", -np.pi, np.pi),
            (np.angle(s_ref.r), axes[2, 2], "twilight", -np.pi, np.pi),
            (e_det_mag, axes[2, 3], "viridis", 0, None),
            (np.abs(obj), axes[2, 4], "viridis", 0, None),
        ]

        for idx, (data, ax, cmap, vmin, vmax) in enumerate(plot_steps):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
            ims.append(im)

            if not cbar_initialized:
                row, col = divmod(idx, 5)
                ax.set_title(titles[row][col], pad=10)  # Added title padding

                # Append colorbar to the right of the image
                divider = make_axes_locatable(ax)
                cax = divider.append_axes(
                    "right", size="5%", pad=0.15
                )  # Increased pad between plot and cbar
                cb = fig.colorbar(im, cax=cax)

                cb.update_ticks()

                ax.set_xticks([])
                ax.set_yticks([])

        frames.append(ims)
        cbar_initialized = True

    # Use rect to prevent the suptitle/titles from clipping at the very top
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    anim = ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=0)
    anim.save("figures/anim.mp4", dpi=200, writer="ffmpeg", fps=10)
    plt.show()


if __name__ == "__main__":
    generate_visual_simulation()
