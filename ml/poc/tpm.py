import matplotlib.pyplot as plt
import numpy as np
import torch

from zern import ZernikeAberration


def generate_grf_phase_torch(
    N: int,
    pupil_N: int | None = None,
    alpha: float = 3.67,
    rms_rad: float = 2.0,
    device: str = "cpu",
) -> torch.Tensor:
    if pupil_N is None:
        pupil_N = 2 * N

    kx = torch.fft.fftfreq(pupil_N, device=device)
    ky = torch.fft.fftfreq(pupil_N, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")
    K = torch.sqrt(KX**2 + KY**2)
    K[0, 0] = 1e-10

    psd = K ** (-alpha / 2.0)
    psd[0, 0] = 0.0

    white_noise = torch.randn(pupil_N, pupil_N, dtype=torch.complex64, device=device)
    field = torch.fft.ifft2(white_noise * psd).real

    # NA=1 Pupil aperture mask (radius = N/2)
    cy, cx = pupil_N // 2, pupil_N // 2
    r_max = N // 2
    coords = torch.arange(pupil_N, device=device)
    y_idx = coords.view(-1, 1)
    x_idx = coords.view(1, -1)
    mask = (((y_idx - cy) ** 2 + (x_idx - cx) ** 2) <= r_max**2).to(dtype=torch.float32)

    # Standardize RMS specifically inside the active pupil aperture
    pupil_pixels = field[mask > 0]
    field = field - pupil_pixels.mean()
    std_val = pupil_pixels.std()

    if std_val > 1e-8:
        field = (field / std_val) * rms_rad

    return field * mask


def generate_composite_sample(
    N: int = 32,
    zern_n: int = 5,
    zern_weight: float = 0.7,
    grf_weight: float = 0.3,
    seed: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    ab_gen = ZernikeAberration(N, zern_n=zern_n)
    num_coeffs = ab_gen.num_coefficients

    j_indices = torch.arange(num_coeffs, dtype=torch.float32)
    radial_orders = torch.floor((-1.0 + torch.sqrt(1.0 + 8.0 * j_indices)) / 2.0)
    kolmogorov_weights = (radial_orders + 1.0) ** (-5 / 6)

    # Scale Zernike coefficients into continuous radian magnitudes
    c_zern = (
        (torch.rand(num_coeffs) * 2 - 1)
        * kolmogorov_weights
        * torch.pi
        / kolmogorov_weights[0]
    )

    theta_zern = ab_gen(c_zern)[N // 2 : N // 2 + N, N // 2 : N // 2 + N]
    mask = torch.where(torch.isnan(theta_zern), 0, 1)

    # If ab_gen returns complex phasors exp(i*theta), unwrap phase to keep it continuous
    if torch.is_complex(theta_zern):
        theta_zern = torch.angle(theta_zern)

    pupil_N = theta_zern.shape[-1]

    alpha = float(torch.empty(1).uniform_(2.5, 5.0))
    rms = float(torch.empty(1).uniform_(3.0, 6.0))
    theta_grf = generate_grf_phase_torch(N, pupil_N=pupil_N, alpha=alpha, rms_rad=rms)

    # Blend continuous phase fields
    theta_total = zern_weight * theta_zern + grf_weight * theta_grf

    return (
        theta_zern.numpy(),
        theta_grf.numpy(),
        theta_total.numpy(),
        alpha,
        rms,
        mask.numpy(),
        pupil_N,
    )


if __name__ == "__main__":
    N = 32
    num_samples = 4

    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 11))
    fig.suptitle(
        "Phase Map Synthesis: Zernike + GRF Composite Fields (NA=1 Pupil)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    col_titles = [
        "Low-Order Zernike\n(Kolmogorov Decay)",
        "GRF Component\n(Random α, RMS)",
        "Combined Phase Field\nθ = 0.7·Zern + 0.3·GRF",
        "Wrapped Phase\nangle(exp(iθ))",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold")

    for i in range(num_samples):
        (
            theta_zern,
            theta_grf,
            theta_total,
            alpha,
            rms,
            mask,
            pupil_N,
        ) = generate_composite_sample(
            N=N, zern_n=5, zern_weight=0.7, grf_weight=0.3, seed=100 + i
        )

        # Mask regions outside the NA=1 pupil aperture
        zern_plot = np.where(mask > 0, theta_zern, np.nan)
        grf_plot = np.where(mask > 0, theta_grf, np.nan)
        total_plot = np.where(mask > 0, theta_total, np.nan)
        wrapped_plot = np.where(mask > 0, np.angle(np.exp(1j * theta_total)), np.nan)

        # Subplot 1: Zernike Component
        im0 = axes[i, 0].imshow(zern_plot, cmap="viridis", interpolation="nearest")
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=11, fontweight="bold")
        fig.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Subplot 2: GRF Component
        im1 = axes[i, 1].imshow(grf_plot, cmap="magma", interpolation="nearest")
        axes[i, 1].text(
            2,
            pupil_N - 4,
            f"α={alpha:.2f}\nRMS={rms:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.5),
        )
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Subplot 3: Combined Continuous Phase
        im2 = axes[i, 2].imshow(total_plot, cmap="cividis", interpolation="nearest")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Subplot 4: Wrapped Complex Phasor Phase
        im3 = axes[i, 3].imshow(
            wrapped_plot,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="nearest",
        )
        fig.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    plt.show()
