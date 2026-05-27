import numpy as np
from config import config
from scipy.signal import fftconvolve


def get_dk_mapping(N):
    """
    Algorithm 2: Pre-compute the flat 2N×2N indices for the Δk mapping.

    For output pixel i = ky_out*N + kx_out and input channel j = ky_in_idx*N + kx_in_idx,
    the Δk = k_out - k_in lands at:
        row = (ky_out + N - ky_in_idx) % 2N
        col = (kx_out + N - kx_in_idx) % 2N
    matching exactly what cass.py produces with pad + np.roll.

    Returns mapping of shape (N*N, N*N), dtype uint32.
    """
    pad_val = N // 2

    # Output pixel positions in the padded 2N×2N grid (after np.pad with pad_val).
    # np.indices gives (row=ky, col=kx) in C-order, matching R_k layout:
    #   R_k.reshape(N,N,N,N)[ky_out, kx_out, ky_in, kx_in]
    r_ones, c_ones = np.indices((N, N))
    r_ones = (r_ones + pad_val).flatten()  # ky_out + pad_val, C-order
    c_ones = (c_ones + pad_val).flatten()  # kx_out + pad_val, C-order

    # Input channel k-indices: ky_in_idx and kx_in_idx run 0..N-1,
    # representing actual centered k-values (ky_in_idx - pad_val).
    # indexing='ij' keeps (ky, kx) row-major, consistent with C-style flatten.
    ky_range = np.arange(-pad_val, pad_val)
    kx_range = np.arange(-pad_val, pad_val)
    KY, KX = np.meshgrid(ky_range, kx_range, indexing="ij")
    shift_y = KY.flatten()  # = ky_in_idx - pad_val for channel j
    shift_x = KX.flatten()  # = kx_in_idx - pad_val for channel j

    # Target flat index in the 2N×2N Δk grid:
    #   row = (ky_out + pad_val - (ky_in_idx - pad_val)) % 2N
    #       = (ky_out + 2*pad_val - ky_in_idx) % 2N
    #       = (ky_out + N - ky_in_idx) % 2N   (since 2*pad_val = N for even N)
    # This matches cass.py: np.roll by (pad_val - y, pad_val - x) on the
    # (ky_out + pad_val, kx_out + pad_val)-positioned padded block.
    rows = (r_ones[:, None] - shift_y[None, :]) % (2 * N)
    cols = (c_ones[:, None] - shift_x[None, :]) % (2 * N)

    mapping = rows * (2 * N) + cols
    return mapping.astype(np.uint32)


def apply_T(v, Rk_flat, mapping, N, dk_mask_flat=None):
    """
    Computes (R̃_Δk · v): accumulate Rk_flat[i,j]*v[j] at Δk index mapping[i,j].

    (R̃_Δk · v)[Δk] = Σ_{k_in} R̃(k_out=Δk+k_in; k_in) * v[k_in]
    """
    res = np.zeros(4 * N * N, dtype=complex)
    contributions = Rk_flat * v[None, :]  # (N², N²)
    np.add.at(res, mapping.ravel(), contributions.ravel())

    if dk_mask_flat is not None:
        res *= dk_mask_flat

    return res


def apply_TH(y, Rk_flat, mapping, N, dk_mask_flat=None):
    """
    Computes (R̃_Δk^H · y): adjoint/Hermitian transpose.

    (R̃_Δk^H · y)[k_in=j] = Σ_{k_out} conj(R̃(k_out; k_in)) * y[Δk_{k_out,k_in}]
    """
    if dk_mask_flat is not None:
        y = y * dk_mask_flat
    y_vals = y[mapping]  # (N², N²)
    res = np.sum(np.conj(Rk_flat) * y_vals, axis=0)  # sum over output index i
    return res


def power_iteration(Rk_flat, mapping, N, max_PI_num, clipping, dk_mask_flat=None):
    """
    Algorithm 3: Find the dominant right singular vector of R̃_Δk.

    R̃_Δk ≈ ẽ_o · p_in^T  (rank-1 approximation, ignoring output aberration).

    The dominant right singular vector of (ẽ_o · p_in^T) is p_in* = e^{-iθ_in},
    because (T^H T) · p_in* = ||ẽ_o||² · N² · p_in* (eigenvalue equation).

    Power iteration (T^H T)^n v / |(T^H T)^n v| therefore converges to
    e^{-iθ_in}, NOT e^{+iθ_in}.  The caller must NOT conjugate the result
    before applying it as a correction (see class_algorithm below).
    """
    m = N * N
    v = np.ones(m, dtype=complex) / np.sqrt(m)

    for _ in range(max_PI_num):
        tmp = apply_T(v, Rk_flat, mapping, N, dk_mask_flat)  # R̃_Δk @ v
        v = apply_TH(tmp, Rk_flat, mapping, N, dk_mask_flat)  # R̃_Δk^H @ (R̃_Δk @ v)
        # v *= clipping
        v = v / (np.abs(v) + 1e-12)  # phase-only normalisation

    return v  # ≈ e^{-iθ_in}


# def image_reconstruction(Rk_flat, mapping, N, th=0.01):
#     """
#     Algorithm 1: Accumulate all input channels in Δk space, then IFFT.
#     Includes the OTF deconvolution step from MATLAB's RM_get_image.m to
#     normalize the Δk overlaps.
#     """
#     # 1. Accumulate image in k-space
#     image_k = np.zeros(4 * N * N, dtype=complex)
#     np.add.at(image_k, mapping.ravel(), Rk_flat.ravel())
#     image_k = image_k.reshape(2 * N, 2 * N)
#
#     # 2. Reconstruct the raw (distorted/unnormalized) image
#     # The factor of 4 matches the MATLAB code's scaling
#     img_raw = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image_k))) * 4.0
#
#     # 3. Build the Optical Transfer Function (OTF)
#     kx = np.arange(-N, N)
#     ky = np.arange(-N, N)
#     KX, KY = np.meshgrid(kx, ky)
#
#     # Circular mask representing the Numerical Aperture
#     mask = (KX**2 / (N / 2) ** 2 + KY**2 / (N / 2) ** 2) < 1.0
#     mask = mask.astype(float)
#
#     # The overlap area of two shifted pupils is the auto-convolution of the mask
#     H = fftconvolve(mask, mask, mode="same")
#
#     # Normalize the OTF
#     H = H / np.max(H)
#
#     # Apply a threshold to prevent division-by-zero or blowing up noise
#     H[H < th] = 1.0
#
#     # 4. Deconvolve in k-space and IFFT
#     image_k_deconv = image_k / H
#     img_deconv = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image_k_deconv)))
#
#     # Return BOTH the raw and deconvolved images
#     return img_raw, img_deconv


def image_reconstruction(Rk_flat, mapping, N, th=0.01):
    """
    Algorithm 1: Accumulate all input channels in Δk space, then IFFT.
    Includes the OTF deconvolution step from MATLAB's RM_get_image.m to
    normalize the Δk overlaps.
    """
    # 1. Accumulate image in k-space
    image_k = np.zeros(4 * N * N, dtype=complex)
    np.add.at(image_k, mapping.ravel(), Rk_flat.ravel())
    image_k = image_k.reshape(2 * N, 2 * N)

    # 2. Reconstruct the raw (distorted/unnormalized) image
    img_raw = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image_k))) * 4.0

    # 3. Build the Optical Transfer Function (OTF)
    kx = np.arange(-N, N)
    ky = np.arange(-N, N)
    KX, KY = np.meshgrid(kx, ky)

    # Circular mask representing the Numerical Aperture
    mask = (KX**2 / (N / 2) ** 2 + KY**2 / (N / 2) ** 2) < 1.0
    mask = mask.astype(float)

    # --- THE FIX ---
    # Use native FFT power spectrum to guarantee perfect centering at (N, N).
    # This avoids the 1-pixel down/right shift introduced by fftconvolve(mode='same')
    mask_f = np.fft.fft2(np.fft.ifftshift(mask))
    H = np.fft.fftshift(np.fft.ifft2(mask_f * np.conj(mask_f)).real)

    # Normalize the OTF
    H = H / np.max(H)

    # Apply a threshold to prevent division-by-zero or blowing up noise
    H[H < th] = 1.0

    # 4. Deconvolve in k-space and IFFT
    image_k_deconv = image_k / H
    img_deconv = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image_k_deconv)))

    # Return BOTH the raw and deconvolved images
    return img_raw, img_deconv


def class_algorithm(Rk, N, max_iteration_number=5, max_PI_num=10, kfilter=6):
    """
    Algorithm 4: Main CLASS algorithm.

    Key sign convention
    -------------------
    power_iteration() returns v ≈ e^{-iθ} (the conjugate of the aberration
    phasor), because it finds the dominant right singular vector of R̃_Δk which
    for a rank-1 model ẽ_o · p_in^T is p_in* = e^{-iθ_in}.

    Therefore the correction is applied WITHOUT np.conj():
        column j *= input_phase[j]   (= e^{-iθ_in(j)})  ← removes input phase
        row    i *= output_phase[i]  (= e^{-iθ_out(i)}) ← removes output phase

    Matching MATLAB Algorithm 4:
        Rk = Rk .* input_phase(:).';    % no conj
        Rk = output_phase(:) .* Rk;     % no conj

    For output aberration, Rk is transposed so power_iteration treats k_out
    as the "input" dimension.
    """
    Rk_flat = Rk.reshape(N * N, N * N)
    mapping = get_dk_mapping(N)

    ab_in = np.ones(N * N, dtype=complex)
    ab_out = np.ones(N * N, dtype=complex)

    clipping = np.zeros((N, N))
    cx = N // 2
    cy = N // 2
    r = N // 2 - 1
    y_idx, x_idx = np.ogrid[:N, :N]
    clipping[(y_idx - cy) ** 2 + (x_idx - cx) ** 2 <= r**2] = 1.0
    clipping = clipping.flatten()

    dk_mask_flat = None
    if kfilter > 0:
        ky2, kx2 = np.ogrid[-N:N, -N:N]
        dk_mask = ((ky2**2 + kx2**2) > kfilter**2).astype(float)
        dk_mask_flat = dk_mask.flatten()

    import tqdm

    for _ in tqdm.tqdm(range(max_iteration_number)):

        # PI gives e^{-iθ_in}; multiply columns directly
        input_phase = power_iteration(
            Rk_flat, mapping, N, max_PI_num, clipping, dk_mask_flat
        )
        ab_in *= input_phase  # accumulate

        Rk_flat = Rk_flat * input_phase[None, :]  # col j *= e^{-iθ_in(j)}

        # Transpose Rk so k_out becomes the "input" dimension for PI.
        # Reuse the same mapping (NOT mapping.T) — same Δk geometry.
        # PI gives e^{-iθ_out}; multiply rows directly (no conj).
        Rk_trans = Rk_flat.T
        output_phase = power_iteration(
            Rk_trans, mapping, N, max_PI_num, clipping, dk_mask_flat
        )
        ab_out *= output_phase  # accumulate

        Rk_flat = Rk_flat * output_phase[:, None]  # row i *= e^{-iθ_out(i)}

    _, final_image = image_reconstruction(Rk_flat, mapping, N)

    return final_image, ab_in.reshape(N, N), ab_out.reshape(N, N)


if __name__ == "__main__":
    import forward_sim
    import reflection_matrix
    import obj

    N = config.N
    s_in, s_out = forward_sim.simulate_batched()
    R_k = reflection_matrix.generate_R_k(s_in, s_out)
    # R_k = reflection_matrix.RM_fft(R)
    final_image, ab_in, ab_out = class_algorithm(
        R_k, N, max_iteration_number=20, max_PI_num=10, kfilter=6
    )

    print(final_image)

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the target directory exists
    os.makedirs("class_imgs", exist_ok=True)

    # ==========================================
    # Figure 1: Image Comparison (CLASS vs Truth)
    # ==========================================
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))

    axs1[0].imshow(np.abs(final_image))
    axs1[0].set_title("Reconstructed image")

    axs1[1].imshow(np.abs(obj.obj))
    axs1[1].set_title("Ground-truth object")

    fig1.tight_layout()
    fig1.savefig(os.path.join("class_imgs", "image_comparison.png"), dpi=150)
    plt.show()

    # ==========================================
    # Figure 2: Phase Aberration Comparison
    # ==========================================
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))

    # Top Row: Estimated
    axs2[0, 0].imshow(-np.angle(ab_in), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs2[0, 0].set_title("Input aberration (estimated θ_in)")

    axs2[0, 1].imshow(-np.angle(ab_out), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs2[0, 1].set_title("Output aberration (estimated θ_out)")

    # Bottom Row: Ground Truth
    axs2[1, 0].imshow(
        np.angle(forward_sim.input_abberations),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs2[1, 0].set_title("True input aberration")

    axs2[1, 1].imshow(
        np.angle(forward_sim.output_abberations),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs2[1, 1].set_title("True output aberration")

    fig2.tight_layout()
    fig2.savefig(os.path.join("class_imgs", "phase_comparison.png"), dpi=150)
    plt.show()
