import numpy as np
import torch
from skimage.draw import random_shapes
from torch.utils.data import Dataset
from forward import Simulation
from zern import ZernikeAberration


class ObjDataset(Dataset):
    def __init__(self, N: int, size: int, seed: int = 42):
        self.N = N
        self.size = size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # thread-safe
        local_seed = self.seed + idx
        local_rng = np.random.default_rng(local_seed)

        image = random_shapes(
            image_shape=(self.N, self.N),
            max_shapes=20,
            min_shapes=5,
            shape=None,
            min_size=0,
            max_size=min(60, self.N // 2),
            channel_axis=None,
            rng=local_rng,
        )[0]

        # print(image.shape)

        image = image.astype(np.float32)

        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-8)

        image_tensor = torch.from_numpy(image)

        return image_tensor


class RMDataset(Dataset):
    def __init__(self, N: int, size: int, zern_n: int, seed: int = 42):
        self.N = N
        self.size = size
        self.seed = seed
        self.obj_dataset = ObjDataset(N, size, seed)

        self.simulation = Simulation(N)
        self.ab_gen = ZernikeAberration(N, zern_n=zern_n)
        self.coeff_count = self.ab_gen.num_coefficients

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        obj = self.obj_dataset[idx]
        # print("obj.shape", obj.shape)

        c_in = torch.rand(self.coeff_count) * 2 - 1
        c_out = torch.rand(self.coeff_count) * 2 - 1

        return c_in, c_out, obj
