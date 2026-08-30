"""
Crop reflection-matrix .mat files from 88x88 to 32x32 in r-space.

For each Rkk_*.mat in data/data/:
  1. Load the (Nx*Ny, Nx*Ny) k-space reflection matrix 'Rkk' -- dims 0,1 =
     k_in, dims 2,3 = k_out once reshaped to (N,N,N,N).
  2. Transform to r-space with RM_ifft.
  3. Center-crop both the r_in and r_out spatial axes from N=88 to
     N_CROP=32.
  4. Transform the cropped field back to k-space with RM_fft (N=32).
  5. Save as Rkk_*.mat in data/data32/, with Nx/Ny updated to 32.

Assumptions (check these against your actual setup):
  - The 88x88 grid is centered, so a symmetric center crop selects the
    physically meaningful ROI.
  - `carrier` / `wavelength_um` / `NA` are carried through unchanged, since
    they describe the physical setup rather than the grid sampling. If
    `carrier` (currently [388, 388]) is defined relative to some
    Nx/Ny-dependent indexing, it likely needs re-referencing for the
    cropped 32x32 grid -- worth double-checking before trusting it.
  - `rm` (RM_fft/RM_ifft) is importable as a sibling module -- adjust the
    import if your layout differs (e.g. `from ml import rm as rm_pt` if run
    from the repo root instead of ml/).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

import rm as rm_pt  # RM_fft / RM_ifft -- adjust import path if needed

SRC_DIR = Path("data/data")
DST_DIR = Path("data/data32")
N_CROP = 32


def crop_rkk(Rkk: np.ndarray, N: int, n_crop: int) -> np.ndarray:
    assert Rkk.shape == (N * N, N * N), f"unexpected shape {Rkk.shape} for N={N}"

    Rkk = np.ascontiguousarray(Rkk)
    # complex128 internally through the two FFT round-trips, for precision
    Rkk_pt = torch.from_numpy(Rkk).to(torch.complex128)

    # k-space -> r-space
    Rrr_pt = rm_pt.RM_ifft(Rkk_pt, N)

    # (N*N, N*N) -> (N, N, N, N): dims 0,1 = r_in, dims 2,3 = r_out
    Rrr_4d = Rrr_pt.reshape(N, N, N, N)

    # symmetric center crop on both the r_in and r_out spatial pairs
    start = (N - n_crop) // 2
    end = start + n_crop
    Rrr_4d_cropped = Rrr_4d[start:end, start:end, start:end, start:end]

    # back to (n_crop*n_crop, n_crop*n_crop) and forward to k-space at the
    # new, smaller N
    Rrr_cropped = Rrr_4d_cropped.reshape(n_crop * n_crop, n_crop * n_crop)
    Rkk_cropped_pt = rm_pt.RM_fft(Rrr_cropped, n_crop)

    return Rkk_cropped_pt.numpy().astype(np.complex64)  # match original dtype


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(SRC_DIR.glob("*.mat"))
    if not mat_files:
        print(f"No .mat files found in {SRC_DIR}")
        return

    for path in mat_files:
        data = loadmat(path)
        Nx = int(data["Nx"].item())
        Ny = int(data["Ny"].item())
        assert Nx == Ny, f"{path.name}: expected square grid, got Nx={Nx}, Ny={Ny}"
        N = Nx

        if N == N_CROP:
            print(f"{path.name}: already {N_CROP}x{N_CROP}, copying as-is")
            Rkk_cropped = data["Rkk"]
        else:
            Rkk_cropped = crop_rkk(data["Rkk"], N, N_CROP)

        out = {
            "Rkk": Rkk_cropped,
            "Nx": np.array([[N_CROP]]),
            "Ny": np.array([[N_CROP]]),
            "carrier": data["carrier"],
            "wavelength_um": data["wavelength_um"],
            "NA": data["NA"],
        }

        out_path = DST_DIR / path.name
        savemat(out_path, out)
        print(f"{path.name}: {N}x{N} -> {N_CROP}x{N_CROP}  saved to {out_path}")


if __name__ == "__main__":
    main()
