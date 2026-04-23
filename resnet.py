# resnet.py
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNetEstimator(nn.Module):
    def __init__(self, n_input, coeff_count):
        super().__init__()
        # We treat Real and Imag as 2 input channels
        # Reshape (N, N, N, N) -> (N^2, N^2)
        self.initial_conv = nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(32, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)

        # Global Pooling makes the model N-agnostic
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 2 * coeff_count)

    def _make_layer(self, in_planes, out_planes, blocks):
        layers = []
        layers.append(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
        )
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(blocks):
            layers.append(ResBlock(out_planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # x shape: (batch, N, N, N, N) complex
        batch_size = x.shape[0]
        N = x.shape[1]

        # Reshape to (Batch, Channels, N*N, N*N)
        # We split complex into 2 channels
        real = x.real.reshape(batch_size, 1, N * N, N * N)
        imag = x.imag.reshape(batch_size, 1, N * N, N * N)
        x = torch.cat((real, imag), dim=1)

        x = self.relu(self.bn(self.initial_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
