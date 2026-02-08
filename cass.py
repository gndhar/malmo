import forward_sim
from config import config
import numpy as np
import fft


def cass(pos, plot=False):
    k_in = np.zeros((len(pos), config.N, config.N))

    for i, (x, y) in enumerate(pos):
        k_in[i, x, y] = 1.0

    k_out = forward_sim.propagate_k_k(k_in)

    k_out_shifted = np.zeros_like(k_out)
    for i, (x, y) in enumerate(pos):
        k_out_shifted[i] = np.roll(k_out[i], shift=(-x, -y), axis=(0, 1))

    r_out_shifted = fft.ifft2(k_out_shifted)

    final_k_out = np.sum(k_out_shifted, axis=(0,))
    final_r_out = fft.ifft2(final_k_out)

    if not plot:
        return final_r_out, final_k_out

    import matplotlib.pyplot as plt

    for i, (x, y) in enumerate(pos):
        plt.subplot(len(pos), 4, 4 * i + 1)
        plt.title(f"{x:.2f}, {y:.2f}")
        plt.imshow(k_in[i])

        plt.subplot(len(pos), 4, 4 * i + 2)
        plt.imshow(np.abs(k_out[i]))

        plt.subplot(len(pos), 4, 4 * i + 3)
        plt.imshow(np.abs(k_out_shifted[i]))

        plt.subplot(len(pos), 4, 4 * i + 4)
        plt.imshow(np.abs(r_out_shifted[i]))

    plt.show()

    plt.subplot(2, 2, 1)
    plt.imshow(np.abs(final_k_out))
    plt.subplot(2, 2, 2)
    plt.imshow(np.angle(final_k_out))
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(final_r_out))
    plt.subplot(2, 2, 4)
    plt.imshow(np.angle(final_r_out))
    plt.show()
    return final_r_out, final_k_out


if __name__ == "__main__":
    N = [i for i in range(1, 1000, 100)]

    def random_point():
        return np.random.randint(0, config.N), np.random.randint(0, config.N)

    def cass_img(n: int):
        return np.abs(cass([random_point() for _ in range(n)], plot=False)[0])

    def scale_match(img, img2):
        return img * np.mean(img2) / np.mean(img)

    def cass_img_intensity(n: int):
        return np.sum(cass_img(n))

    def img_error(img1: np.ndarray, img2: np.ndarray) -> float:
        return np.mean(np.abs(np.abs(img2) - np.abs(img1)), dtype=float)

    def img_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        return np.abs(img1) - np.abs(img2)

    from tqdm import tqdm

    imgs = [scale_match(cass_img(n), forward_sim.obj) for n in tqdm(N)]

    import matplotlib.pyplot as plt

    for i, img in enumerate(imgs, start=1):
        plt.subplot(2, len(imgs) // 2, i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Samples: {N[i-1]}")
    plt.show()

    errs = [img_diff(np.abs(img), forward_sim.obj) for img in imgs]

    vmin = np.min(errs)
    vmax = np.max(errs)
    for i, img in enumerate(errs, start=1):
        plt.subplot(2, len(imgs) // 2, i)
        plt.imshow(img, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title(f"Samples: {N[i-1]}")
        plt.colorbar()
    plt.show()

    err_val = [img_error(img, forward_sim.obj) for img in imgs]
    plt.plot(N, err_val, label="absolute error")
    plt.yscale("log", base=10)
    plt.legend()
    plt.show()
