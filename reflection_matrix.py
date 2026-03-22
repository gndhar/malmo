import forward_sim
from forward_sim import Signal
import numpy as np
from config import config

N = config.N


def generate_R2(s_in: Signal, s_out: Signal):
    r_in = s_in.r
    r_out = s_out.r

    idxs = [(x, y) for x in range(N) for y in range(N)]

    A = np.column_stack([r_in[x, y].flatten(order="F") for x, y in idxs])
    B = np.column_stack([r_out[x, y].flatten(order="F") for x, y in idxs])

    R = B @ A.conj().T
    return R, A, B


def RM_fft2(M_xin_xout):
    """
    Calculates the Fourier transform of a position domain reflection matrix.

    Parameters:
        M_xin_xout: 2D array (Nx*Ny, Nx*Ny)
        Nx: Number of sampling points in x
        Ny: Number of sampling points in y
    """

    Nx = Ny = config.N
    # 1. Dimension Check
    if M_xin_xout.shape[0] != Nx * Ny:
        raise ValueError("Check matrix dimensions!!")

    # 2. Process Output Coordinates (Rows)
    # Reshape using Fortran order to match MATLAB's (Ny, Nx, [])
    # -1 tells NumPy to calculate the remaining dimension automatically
    M_temp = np.reshape(M_xin_xout, (Ny, Nx, -1), order="F")

    # Apply fftshift on the first two axes (spatial dimensions)
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 2D FFT on the first two axes
    M_temp = np.fft.fft2(M_temp, axes=(0, 1))

    # Shift back
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 3. Transpose to switch focus to Input Coordinates (Columns)
    # Reshape to (Ny*Nx, []) and Transpose
    M_temp = np.reshape(M_temp, (Ny * Nx, -1), order="F").T

    # 4. Process Input Coordinates
    M_temp = np.reshape(M_temp, (Ny, Nx, -1), order="F")
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # Note: Using IFFT2 as per the original MATLAB logic
    M_temp = np.fft.ifft2(M_temp, axes=(0, 1))

    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 5. Final Flatten and Transpose back to original orientation
    M_kin_kout = np.reshape(M_temp, (Ny * Nx, -1), order="F").T

    return M_kin_kout


def RM_fft(M_xin_xout):
    Nx = Ny = config.N
    if M_xin_xout.shape[0] != Nx * Ny:
        raise ValueError("Check matrix dimensions!!")

    # 2. Process Output Coordinates (Rows)
    # Reshape using Fortran order to match MATLAB's (Ny, Nx, [])
    # -1 tells NumPy to calculate the remaining dimension automatically
    M_temp = np.reshape(M_xin_xout, (Ny, Nx, -1))

    # Apply fftshift on the first two axes (spatial dimensions)
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 2D FFT on the first two axes
    M_temp = np.fft.fft2(M_temp, axes=(0, 1))

    # Shift back
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 3. Transpose to switch focus to Input Coordinates (Columns)
    # Reshape to (Ny*Nx, []) and Transpose
    M_temp = np.reshape(M_temp, (Ny * Nx, -1)).T

    # 4. Process Input Coordinates
    M_temp = np.reshape(M_temp, (Ny, Nx, -1))
    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # Note: Using IFFT2 as per the original MATLAB logic
    M_temp = np.fft.ifft2(M_temp, axes=(0, 1))

    M_temp = np.fft.fftshift(M_temp, axes=(0, 1))

    # 5. Final Flatten and Transpose back to original orientation
    M_kin_kout = np.reshape(M_temp, (Ny * Nx, -1)).T

    return M_kin_kout


def generate_R(s_in: Signal, s_out: Signal):
    r_in = s_in.r
    r_out = s_out.r

    idxs = [(y, x) for y in range(N) for x in range(N)]

    A = np.column_stack([r_in[y, x].flatten() for y, x in idxs])
    B = np.column_stack([r_out[y, x].flatten() for y, x in idxs])

    R = B @ A.conj().T
    return R, A, B


def generate_Rk(s_in: Signal, s_out: Signal):
    r_in = s_in.k
    r_out = s_out.k

    idxs = [(y, x) for y in range(N) for x in range(N)]

    A = np.column_stack([r_in[y, x].flatten() for y, x in idxs])
    B = np.column_stack([r_out[y, x].flatten() for y, x in idxs])

    R = B @ A.conj().T
    return R, A, B


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import fft

    def compare(A, B):
        return {
            "true": np.all(A == B),
            "approx": np.allclose(A, B, atol=1e-5),
            "min_diff": np.max(np.abs(A - B)),
        }

    def is_unitary(A):
        return compare(A @ A.conj().T, np.eye(A.shape[0]))

    s_in, s_out = forward_sim.simulate()
    u_in = np.zeros((config.N, config.N))
    x = 31
    y = 12
    u_in[y, x] = 1.0

    # C-style (Python default, numpy by extension)
    R, A, B = generate_R(s_in, s_out)
    print(is_unitary(A))
    Rk, Ak, Bk = generate_Rk(s_in, s_out)
    print(is_unitary(Ak))
    R_k = RM_fft(R)
    print(compare(R_k, Rk))
    v_k = (R_k @ u_in.flatten()).reshape(config.N, config.N)

    plt.imshow(np.abs(fft.ifft2(v_k)))
    plt.colorbar()
    plt.show()

    v_out = (R @ u_in.flatten()).reshape(config.N, config.N)

    # Fortran (MATLAB) style
    R2, A2, B2 = generate_R2(s_in, s_out)
    R_k2 = RM_fft2(R2)
    v_k2 = (R_k2 @ u_in.flatten(order="F")).reshape(config.N, config.N, order="F")

    plt.imshow(np.abs(fft.ifft2(v_k2)))
    plt.colorbar()
    plt.show()

    v_out2 = (R2 @ u_in.flatten(order="F")).reshape(config.N, config.N, order="F")

    print(np.all(v_k == v_k2))  # check first column equal
    print(np.all(v_out == v_out2))  # check both values equal
    print(np.all(R[:, y * config.N + x].reshape(config.N, config.N) == v_out2))

    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(u_in))
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(v_out))
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(v_out2))
    plt.colorbar()

    plt.show()

    k_out1 = R_k.reshape(config.N, config.N, config.N, config.N)[:, :, y, x]
    k_out2 = (R_k @ u_in.flatten()).reshape(config.N, config.N)
    print(np.all(k_out1 == k_out2))
