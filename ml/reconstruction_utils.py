import torch


def get_dk_mapping(N, device="cpu"):
    """
    Pre-compute the flat 2N×2N indices for the Δk mapping.

    Returns mapping of shape (N*N, N*N), dtype torch.long (required for
    PyTorch indexing/scatter operations).
    """
    pad_val = N // 2

    # Output pixel positions in the padded 2N×2N grid
    # torch.meshgrid with indexing='ij' matches np.indices C-order
    y_ones, x_ones = torch.meshgrid(
        torch.arange(N, device=device), torch.arange(N, device=device), indexing="ij"
    )
    r_ones = (y_ones + pad_val).flatten()
    c_ones = (x_ones + pad_val).flatten()

    # Input channel k-indices
    ky_range = torch.arange(-pad_val, pad_val, device=device)
    kx_range = torch.arange(-pad_val, pad_val, device=device)
    KY, KX = torch.meshgrid(ky_range, kx_range, indexing="ij")

    shift_y = KY.flatten()
    shift_x = KX.flatten()

    # Target flat index in the 2N×2N Δk grid
    rows = (r_ones[:, None] - shift_y[None, :]) % (2 * N)
    cols = (c_ones[:, None] - shift_x[None, :]) % (2 * N)

    mapping = rows * (2 * N) + cols

    # Must be torch.int64 (long) to be used as indices in scatter_add_
    return mapping.to(torch.long)


def image_reconstruction(Rk_flat, mapping, N, th=0.01):
    """
    Accumulate all input channels in Δk space, then IFFT.
    Handles batched inputs natively.

    Args:
        Rk_flat: Tensor of shape (B, N*N, N*N) or (N*N, N*N)
        mapping: Tensor of shape (N*N, N*N) from get_dk_mapping
        N: Integer representing the grid size base
        th: Threshold for OTF normalisation

    Returns:
        img_raw: Tensor of shape (B, 2N, 2N)
        img_deconv: Tensor of shape (B, 2N, 2N)
    """
    device = Rk_flat.device

    # Standardize to 3D (B, N*N, N*N) to handle batching smoothly
    is_unbatched = Rk_flat.dim() == 2
    if is_unbatched:
        Rk_flat = Rk_flat.unsqueeze(0)

    B = Rk_flat.shape[0]

    # 1. Accumulate image in k-space
    # Flatten the spatial dimensions: (B, N^4)
    Rk_flat_1d = Rk_flat.reshape(B, -1)

    # Broadcast mapping to match batch size: (B, N^4)
    mapping_batched = mapping.view(1, -1).expand(B, -1)

    # Initialize target tensor and scatter add
    image_k_1d = torch.zeros((B, 4 * N * N), dtype=Rk_flat.dtype, device=device)
    image_k_1d.scatter_add_(1, mapping_batched, Rk_flat_1d)

    # Reshape back to 2D grid: (B, 2N, 2N)
    image_k = image_k_1d.view(B, 2 * N, 2 * N)

    # 2. Reconstruct the raw (distorted/unnormalized) image
    # Use dim=(-2, -1) to ensure we only transform spatial dimensions, not batch
    img_raw_k = torch.fft.fftshift(image_k, dim=(-2, -1))
    img_raw_ifft = torch.fft.ifft2(img_raw_k, dim=(-2, -1))
    img_raw = torch.fft.ifftshift(img_raw_ifft, dim=(-2, -1)) * 4.0

    # 3. Build the Optical Transfer Function (OTF)
    # This is static for a given N, so we build it without a batch dimension
    kx = torch.arange(-N, N, device=device)
    ky = torch.arange(-N, N, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")

    # Circular mask representing the Numerical Aperture
    mask = ((KX**2 / (N / 2) ** 2 + KY**2 / (N / 2) ** 2) < 1.0).float()

    # Native FFT power spectrum to guarantee perfect centering
    mask_f = torch.fft.fft2(torch.fft.ifftshift(mask, dim=(-2, -1)), dim=(-2, -1))
    H_unshifted = torch.fft.ifft2(mask_f * torch.conj(mask_f), dim=(-2, -1)).real
    H = torch.fft.fftshift(H_unshifted, dim=(-2, -1))

    # Normalize the OTF
    H = H / torch.max(H)

    # Apply a threshold
    H = torch.where(H < th, torch.tensor(1.0, dtype=H.dtype, device=device), H)

    # 4. Deconvolve in k-space and IFFT
    # Broadcasting takes care of dividing (B, 2N, 2N) by (2N, 2N)
    image_k_deconv = image_k / H

    img_deconv_k = torch.fft.fftshift(image_k_deconv, dim=(-2, -1))
    img_deconv_ifft = torch.fft.ifft2(img_deconv_k, dim=(-2, -1))
    img_deconv = torch.fft.ifftshift(img_deconv_ifft, dim=(-2, -1))

    # Return unbatched if the input lacked a batch dimension
    if is_unbatched:
        return img_raw.squeeze(0), img_deconv.squeeze(0)

    return img_raw, img_deconv
