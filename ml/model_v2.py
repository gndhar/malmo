from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    coeff_count: int
    base_width: int = 32
    blocks_per_stage: Tuple[int, ...] = (2, 2)
    num_stages: int = 2
    stage_width_mult: int = 2  # width doubles each stage
    norm_groups: int = 8  # GroupNorm groups
    use_coord_conv: bool = False
    use_sum_diff_basis: bool = False  # rotate to (x1+x2, x1-x2) before stem
    stem_kernel_size: int = 7


def make_norm(num_channels: int, groups: int) -> nn.Module:
    # GroupNorm needs num_channels % groups == 0; fall back gracefully.
    g = groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResBlock(nn.Module):
    def __init__(self, planes: int, norm_groups: int):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.norm1 = make_norm(planes, norm_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.norm2 = make_norm(planes, norm_groups)

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        return self.relu(out)


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized (row, col) coordinate channels in [-1, 1]."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_y, grid_x], dim=0)  # (2, H, W)
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)  # (B, 2, H, W)
    return torch.cat([x, grid], dim=1)


def rotate_45(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the spatial grid by 45 degrees so that the x1==x2 diagonal
    (memory-effect direction) becomes axis-aligned. Uses bilinear
    resampling via affine_grid/grid_sample; output size matches input.
    """
    b = x.shape[0]
    theta = (
        torch.tensor(
            [[0.7071, -0.7071, 0.0], [0.7071, 0.7071, 0.0]],
            device=x.device,
            dtype=x.dtype,
        )
        .unsqueeze(0)
        .expand(b, -1, -1)
    )
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


class Model(nn.Module):
    def __init__(self, N: int, coeff_count: int, config: Optional[ModelConfig] = None):
        super().__init__()
        self.N = N
        cfg = config or ModelConfig(coeff_count=coeff_count)
        self.cfg = cfg

        in_ch = 2  # real, imag
        if cfg.use_coord_conv:
            in_ch += 2

        self.norm_in = nn.InstanceNorm2d(
            2, affine=False
        )  # per-sample scale fix for real/imag only

        self.initial_conv = nn.Conv2d(
            in_ch,
            cfg.base_width,
            kernel_size=cfg.stem_kernel_size,
            stride=1,
            padding=cfg.stem_kernel_size // 2,
        )
        self.stem_norm = make_norm(cfg.base_width, cfg.norm_groups)
        self.relu = nn.ReLU(inplace=True)

        stages = []
        in_planes = cfg.base_width
        for i in range(cfg.num_stages):
            out_planes = in_planes if i == 0 else in_planes * cfg.stage_width_mult
            n_blocks = (
                cfg.blocks_per_stage[i]
                if i < len(cfg.blocks_per_stage)
                else cfg.blocks_per_stage[-1]
            )
            stages.append(
                self._make_stage(in_planes, out_planes, n_blocks, cfg.norm_groups)
            )
            in_planes = out_planes
        self.stages = nn.Sequential(*stages)
        self.final_planes = in_planes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_in = nn.Linear(self.final_planes, coeff_count)
        self.fc_out = nn.Linear(self.final_planes, coeff_count)

    @staticmethod
    def _make_stage(in_planes, out_planes, blocks, norm_groups):
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
            make_norm(out_planes, norm_groups),
            nn.ReLU(inplace=True),
        ]
        for _ in range(blocks):
            layers.append(ResBlock(out_planes, norm_groups))
        return nn.Sequential(*layers)

    def forward(self, Rk: torch.Tensor):
        """
        Rk: complex tensor of shape (batch, N, N, N, N), representing
        R(x1, y1, x2, y2). Reshaped to (batch, N^2, N^2) spatial maps,
        matching the "(N,N,N,N) -> (N^2,N^2)" convention from the class doc.
        """
        N2 = self.N * self.N
        batch_size = Rk.shape[0]

        real = Rk.real.reshape(batch_size, 1, N2, N2)
        imag = Rk.imag.reshape(batch_size, 1, N2, N2)
        x = torch.cat((real, imag), dim=1)

        x = self.norm_in(x)  # per-sample normalization of real/imag scale

        if self.cfg.use_sum_diff_basis:
            x = rotate_45(x)

        if self.cfg.use_coord_conv:
            x = add_coord_channels(x)

        x = self.relu(self.stem_norm(self.initial_conv(x)))
        x = self.stages(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        pred_in = self.fc_in(x)
        pred_out = self.fc_out(x)
        return pred_in, pred_out


if __name__ == "__main__":
    # quick smoke test
    N = 16
    coeff_count = 21
    cfg = ModelConfig(coeff_count=coeff_count, use_sum_diff_basis=True)
    model = Model(N=N, coeff_count=coeff_count, config=cfg)
    Rk = torch.randn(2, N, N, N, N, dtype=torch.complex64)
    pred_in, pred_out = model(Rk)
    print("pred_in:", pred_in.shape, "pred_out:", pred_out.shape)
