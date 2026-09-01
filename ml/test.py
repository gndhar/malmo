import torch
import torch.nn.functional as F


def generate_high_depth_grf(
    batch_size: int,
    N: int = 64,
    kernel_size: int = 17,  # Larger kernel to support larger sigma without truncation
    sigma_spatial: float = 2.0,  # Enforces smooth, low-frequency spatial variation
    min_wraps: float = 2.0,
    max_wraps: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generates continuous smooth phase fields with 4 to 6 full 2pi wrap cycles across the grid."""
    # 1. Construct broad 2D Gaussian smoothing filter
    coords = (
        torch.arange(kernel_size, dtype=torch.float32, device=device)
        - (kernel_size - 1) / 2
    )
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    gaussian_kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma_spatial**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # 2. Filter raw noise to create smooth topography
    noise = torch.randn(batch_size, 1, N, N, device=device)
    smoothed = F.conv2d(noise, gaussian_kernel, padding=kernel_size // 2)

    # 3. Standardize spatial field per sample to unit variance
    smoothed_std = smoothed.std(dim=(-2, -1), keepdim=True) + 1e-8
    smooth_normalized = smoothed / smoothed_std

    # 4. Scale peak-to-valley range to match requested 2pi wrap count
    # Peak-to-valley targeted between (min_wraps * 2pi) and (max_wraps * 2pi)
    target_wraps = (
        torch.empty(batch_size, 1, 1, 1, device=device).uniform_(min_wraps, max_wraps)
        * 2
        * torch.pi
    )
    ptv_current = smooth_normalized.amax(
        dim=(-2, -1), keepdim=True
    ) - smooth_normalized.amin(dim=(-2, -1), keepdim=True)

    continuous_phase = (smooth_normalized / ptv_current) * target_wraps
    return continuous_phase.squeeze(1)  # Shape: (batch_size, N, N)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import zern

    N = 64
    num_samples = 16

    # 1. Obtain pupil mask and generate batch of 16 GRF phases
    zern_gen = zern.ZernikeAberration(N=N // 2, zern_n=0)
    mask = zern_gen.pupil_mask

    phases = generate_high_depth_grf(num_samples, N=N)
    angles = torch.angle(torch.exp(1j * phases))  # Wrap continuous phase into [-pi, pi]

    # 2. Setup 4x4 subplot visualization
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("High-Depth GRF Wrapped Phase Maps (4x4 Grid)", fontsize=14, y=0.95)

    for i, ax in enumerate(axes.flat):
        # Apply mask and convert to numpy
        masked_angle = torch.where(mask > 0, angles[i], torch.nan).cpu().numpy()

        im = ax.imshow(masked_angle, cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
        ax.set_title(f"Sample {i+1}", fontsize=9)
        ax.axis("off")

    # 3. Add single unified colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([-torch.pi, 0, torch.pi])
    cbar.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

    plt.show()
