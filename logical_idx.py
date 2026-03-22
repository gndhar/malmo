import numpy as np


def get_dk_logical_index_slow(Nx, Ny):
    """Direct MATLAB translation using explicit loops and matrices."""
    Nky, Nkx = Ny // 2, Nx // 2

    # Meshgrid matches MATLAB's 'xy' default
    kx_range = np.arange(-Nkx, Nkx)
    ky_range = np.arange(-Nky, Nky)
    kx_grid, ky_grid = np.meshgrid(kx_range, ky_range)

    # Flatten to Column-Major ('F') to match MATLAB linear indexing
    kx = kx_grid.flatten("F")
    ky = ky_grid.flatten("F")

    # Matrix size: (4*Ny*Nx) rows by (Ny*Nx) columns
    dk_matrix = np.zeros((4 * Ny * Nx, Nx * Ny), dtype=np.uint8)
    # padarray(ones(Ny, Nx), [Nky, Nkx])
    kmap = np.pad(np.ones((Ny, Nx)), ((Nky, Nky), (Nkx, Nkx)), mode="constant")

    for ii in range(Nx * Ny):
        # circshift shifts circularly. NumPy's roll(A, s) moves element at 0 to s.
        # Shift [-ky, -kx] moves row r to (r-ky)%2Ny
        shifted_kmap = np.roll(kmap, shift=(-ky[ii], -kx[ii]), axis=(0, 1))
        dk_matrix[:, ii] = shifted_kmap.flatten("F")

    # CRITICAL: flatten matrix in 'F' order before finding indices to match MATLAB
    indx_dk = np.flatnonzero(dk_matrix.flatten("F") == 1).astype(np.uint32)
    return indx_dk


def get_dk_logical_index_fast(Nx, Ny):
    """Optimized version using broadcasting (No huge matrices)."""
    Nky, Nkx = Ny // 2, Nx // 2
    kx_range = np.arange(-Nkx, Nkx)
    ky_range = np.arange(-Nky, Nky)
    KX, KY = np.meshgrid(kx_range, ky_range)
    shift_x = KX.flatten("F")
    shift_y = KY.flatten("F")

    # Coordinates of 'ones' in the initial padded kmap
    r_ones, c_ones = np.indices((Ny, Nx))
    r_ones = (r_ones + Nky).flatten("F")
    c_ones = (c_ones + Nkx).flatten("F")

    # Apply all shifts simultaneously using broadcasting
    # final_rows/cols shape: (N_ones, N_shifts)
    final_rows = (r_ones[:, None] - shift_y[None, :]) % (2 * Ny)
    final_cols = (c_ones[:, None] - shift_x[None, :]) % (2 * Nx)

    # Column-major linear index formula: row + col * total_rows
    # grid is (2Ny x 2Nx)
    linear_indices_in_col = final_rows + final_cols * (2 * Ny)

    # MATLAB's find() scans top-to-bottom. After a circular shift,
    # the ones may wrap around. We must sort each column's indices.
    linear_indices_in_col = np.sort(linear_indices_in_col, axis=0)

    # Global linear indexing for dk_matrix (rows=4*Ny*Nx, cols=Nx*Ny)
    M = 4 * Ny * Nx
    col_offsets = np.arange(Nx * Ny) * M
    global_indices = linear_indices_in_col + col_offsets[None, :]

    # Flatten 'F' to preserve col-by-col concatenation
    return global_indices.flatten("F").astype(np.uint32)


def get_dk_logical_index_c_style(N):
    """
    Creates a mapping from R_k (flattened) to total_k_out (linear indices).
    Adheres to C-style (row-major) indexing.
    """
    # 1. Define the target grid dimensions (2N x 2N)
    target_shape = (2 * N, 2 * N)

    # 2. Coordinates of the NxN source block inside the 2Nx2N padded grid
    # Initial position: centered due to pad_val = N // 2
    pad_val = N // 2
    r_src, c_src = np.indices((N, N))
    r_src += pad_val
    c_src += pad_val

    # 3. Define the shifts (x_, y_) for every (x, y) in the NxN input grid
    # Matches your: x_ = (N // 2) - x
    y_range = np.arange(N)
    x_range = np.arange(N)
    Y, X = np.meshgrid(y_range, x_range, indexing="ij")  # 'ij' is C-style friendly

    shift_y = (N // 2) - Y.flatten()
    shift_x = (N // 2) - X.flatten()

    # 4. Calculate shifted positions using broadcasting
    # result shape: (N_pixels_in_block, N_shifts) -> (N^2, N^2)
    final_rows = (r_src.flatten()[:, None] + shift_y[None, :]) % (2 * N)
    final_cols = (c_src.flatten()[:, None] + shift_x[None, :]) % (2 * N)

    # 5. Convert to C-style linear indices: row * total_cols + col
    # No 'F' order needed here; NumPy defaults to 'C'
    linear_indices = final_rows * (2 * N) + final_cols

    return linear_indices.astype(np.uint32)


# --- Verification ---
if __name__ == "__main__":
    from config import config

    Nx = Ny = config.N
    idx_slow = get_dk_logical_index_slow(Nx, Ny)
    idx_fast = get_dk_logical_index_fast(Nx, Ny)
    idx_c = get_dk_logical_index_c_style(config.N)

    print(f"Match: {np.array_equal(idx_slow, idx_fast)}")
    print(f"Match: {np.array_equal(idx_c, idx_fast)}")
    print(idx_slow)
    print(idx_fast)
    print(idx_c)
    print(idx_c.shape)
    # If comparing to MATLAB, remember Python is 0-indexed.
    # Use idx_fast + 1 to get exact MATLAB values.
