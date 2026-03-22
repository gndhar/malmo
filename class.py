import numpy as np
from config import config


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


def apply_T(v, Rk_flat, mapping, N):
    """
    Computes (R̃_Δk · v): accumulate Rk_flat[i,j]*v[j] at Δk index mapping[i,j].

    (R̃_Δk · v)[Δk] = Σ_{k_in} R̃(k_out=Δk+k_in; k_in) * v[k_in]
    """
    res = np.zeros(4 * N * N, dtype=complex)
    contributions = Rk_flat * v[None, :]  # (N², N²)
    np.add.at(res, mapping.ravel(), contributions.ravel())
    return res


def apply_TH(y, Rk_flat, mapping, N):
    """
    Computes (R̃_Δk^H · y): adjoint/Hermitian transpose.

    (R̃_Δk^H · y)[k_in=j] = Σ_{k_out} conj(R̃(k_out; k_in)) * y[Δk_{k_out,k_in}]
    """
    y_vals = y[mapping]  # (N², N²)
    res = np.sum(np.conj(Rk_flat) * y_vals, axis=0)  # sum over output index i
    return res


def power_iteration(Rk_flat, mapping, N, max_PI_num):
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
        tmp = apply_T(v, Rk_flat, mapping, N)  # R̃_Δk @ v
        v = apply_TH(tmp, Rk_flat, mapping, N)  # R̃_Δk^H @ (R̃_Δk @ v)
        v = v / (np.abs(v) + 1e-12)  # phase-only normalisation

    return v  # ≈ e^{-iθ_in}


def image_reconstruction(Rk_flat, mapping, N):
    """
    Algorithm 1: Accumulate all input channels in Δk space, then IFFT.

    Equivalent to the nested pad + np.roll loop in cass.py.
    """
    image_k = np.zeros(4 * N * N, dtype=complex)
    np.add.at(image_k, mapping.ravel(), Rk_flat.ravel())

    image_k = image_k.reshape(2 * N, 2 * N)
    img = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(image_k)))
    return img


def class_algorithm(Rk, N, max_iteration_number=5, max_PI_num=10):
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
    as the "input" dimension.  The SAME mapping is reused (NOT mapping.T) —
    only the data matrix changes.  This matches MATLAB which reuses indx_dk.
    """
    Rk_flat = Rk.reshape(N * N, N * N)
    mapping = get_dk_mapping(N)

    ab_in = np.ones(N * N, dtype=complex)
    ab_out = np.ones(N * N, dtype=complex)

    for _ in range(max_iteration_number):

        # PI gives e^{-iθ_in}; multiply columns directly
        input_phase = power_iteration(Rk_flat, mapping, N, max_PI_num)
        ab_in *= input_phase  # accumulate

        Rk_flat = Rk_flat * input_phase[None, :]  # col j *= e^{-iθ_in(j)}

        # Transpose Rk so k_out becomes the "input" dimension for PI.
        # Reuse the same mapping (NOT mapping.T) — same Δk geometry.
        # PI gives e^{-iθ_out}; multiply rows directly (no conj).
        Rk_trans = Rk_flat.T
        output_phase = power_iteration(Rk_trans, mapping, N, max_PI_num)
        ab_out *= output_phase  # accumulate

        Rk_flat = Rk_flat * output_phase[:, None]  # row i *= e^{-iθ_out(i)}

    final_image = image_reconstruction(Rk_flat, mapping, N)

    return final_image, ab_in.reshape(N, N), ab_out.reshape(N, N)


if __name__ == "__main__":
    import forward_sim
    import reflection_matrix
    import obj

    N = config.N
    s_in, s_out = forward_sim.simulate()
    R, *_ = reflection_matrix.generate_R(s_in, s_out)
    R_k = reflection_matrix.RM_fft(R)
    final_image, ab_in, ab_out = class_algorithm(
        R_k, N, max_iteration_number=20, max_PI_num=10
    )

    import matplotlib.pyplot as plt

    plt.subplot(2, 3, 1)
    plt.imshow(np.abs(final_image))
    plt.title("CLASS image")
    plt.subplot(2, 3, 2)
    plt.imshow(-np.angle(ab_in), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Input aberration (estimated θ_in)")
    plt.subplot(2, 3, 3)
    plt.imshow(-np.angle(ab_out), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    plt.title("Output aberration (estimated θ_out)")
    plt.subplot(2, 3, 4)
    plt.imshow(obj.obj)
    plt.title("Ground-truth object")
    plt.subplot(2, 3, 5)
    plt.imshow(
        np.angle(forward_sim.input_abberations),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.title("True input aberration")
    plt.subplot(2, 3, 6)
    plt.imshow(
        np.angle(forward_sim.output_abberations),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    plt.title("True output aberration")
    plt.tight_layout()
    plt.show()
