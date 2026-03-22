import numpy as np


def fftshift2(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(x, axes=(-2, -1))


def ifftshift2(x: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(x, axes=(-2, -1))


def fft2(x: np.ndarray) -> np.ndarray:
    return fftshift2(np.fft.fft2(ifftshift2(x), norm="ortho"))


def ifft2(x: np.ndarray) -> np.ndarray:
    return fftshift2(np.fft.ifft2(ifftshift2(x), norm="ortho"))
