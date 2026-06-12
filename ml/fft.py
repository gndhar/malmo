import torch


def fftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))


def fft2(x: torch.Tensor) -> torch.Tensor:
    return fftshift2(torch.fft.fft2(ifftshift2(x), dim=(-2, -1), norm="ortho"))


def ifft2(x: torch.Tensor) -> torch.Tensor:
    return fftshift2(torch.fft.ifft2(ifftshift2(x), dim=(-2, -1), norm="ortho"))
