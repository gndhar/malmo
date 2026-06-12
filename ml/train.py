import torch
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, N: int):
        self.N = N
