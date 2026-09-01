import matplotlib.pyplot as plt
import numpy as np


def generate_grf_phase(
    N: int = 32,
    alpha: float = 3.0,
    rms_rad: float = 2.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 2D Gaussian Random Field (GRF) phase screen.

    Parameters:
    -----------
    N : int
        Grid size (N x N).
    alpha : float
        Power-law spectral decay exponent (P(k) ~ k^(-alpha)).
        - alpha = 2.0: Rougher, high spatial frequency fluctuations.
        - alpha = 3.67 (11/3): Kolmogorov atmospheric turbulence.
        - alpha = 4.0: Smooth, low-frequency dominated phase screens.
    rms_rad : float
        Root-mean-square (RMS) phase fluctuation amplitude in radians.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    phi_unwrapped : np.ndarray
        Unwrapped continuous phase screen (radians) within aperture.
    phi_wrapped : np.ndarray
        Wrapped phase screen in range [-pi, pi].
    mask : np.ndarray
        Binary circular pupil aperture mask.
    """
    if seed is not None:
        np.random.seed(seed)

    # Spatial frequency coordinates
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # Avoid division by zero at DC (zero frequency)
    K[0, 0] = 1e-10

    # Power Spectral Density filter: P(k) ~ K^(-alpha/2)
    psd = K ** (-alpha / 2.0)
    psd[0, 0] = 0.0  # Zero DC component (no constant phase offset)

    # Complex Gaussian white noise in Fourier space
    white_noise = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))

    # Filter noise in Fourier domain and transform to spatial domain
    field_fft = white_noise * psd
    field = np.fft.ifft2(field_fft).real

    # Center zero mean
    field -= np.mean(field)

    # Scale to specified RMS phase error (radians)
    std_val = np.std(field)
    if std_val > 0:
        field = (field / std_val) * rms_rad

    # Circular pupil aperture mask
    cy, cx = N // 2, N // 2
    r_max = N // 2 - 1
    y_idx, x_idx = np.ogrid[:N, :N]
    mask = ((y_idx - cy) ** 2 + (x_idx - cx) ** 2 <= r_max**2).astype(float)

    phi_unwrapped = field * mask
    phi_wrapped = np.angle(np.exp(1j * phi_unwrapped)) * mask

    return phi_unwrapped, phi_wrapped, mask


if __name__ == "__main__":
    N = 32
    alphas = [2.0, 3.67, 4.5]  # Rough -> Kolmogorov -> Smooth
    rms_values = [1.5, 3.0, 5.0]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))

    for row_idx, alpha in enumerate(alphas):
        for col_idx, rms in enumerate(rms_values):
            ax = axes[row_idx, col_idx]
            seed = 42 + row_idx * 3 + col_idx

            phi_raw, phi_wrapped, mask = generate_grf_phase(
                N=N, alpha=alpha, rms_rad=rms, seed=seed
            )

            # Mask out region outside pupil for plotting
            phi_plot = np.where(mask > 0, phi_wrapped, np.nan)

            im = ax.imshow(
                phi_plot,
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="nearest",
            )
            ax.set_title(
                f"α = {alpha} (spectral decay)\nRMS = {rms} rad",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Gaussian Random Field (GRF) Phase Screens [-π, π]",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    plt.show()
