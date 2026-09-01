import torch
from torch import nn
import torch.nn.functional as F


class TensorFolder(nn.Module):

    def __init__(self, N: int, eps: float = 1e-8):
        super().__init__()
        self.N: int = N
        self.eps: float = eps

    def forward(self, Rk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Phase Anchoring
        spatial_dims = tuple(range(1, Rk.ndim))
        weighted_sum = torch.sum(Rk * torch.abs(Rk), dim=spatial_dims, keepdim=True)
        ref_phase = torch.angle(weighted_sum)
        Rk = Rk * torch.exp(-1j * ref_phase)

        # RMS normalization: Rk / sqrt(mean(|Rk|^2))
        rms = torch.sqrt(
            torch.mean(torch.abs(Rk) ** 2, dim=spatial_dims, keepdim=True) + self.eps
        )
        Rk = Rk / rms

        batch_size = Rk.shape[0]
        if torch.is_complex(Rk):
            Rk = torch.stack([Rk.real, Rk.imag], dim=1)

        Rk = Rk.reshape(batch_size, 2, self.N, self.N, self.N, self.N)
        branch_1 = Rk.permute(0, 1, 4, 5, 2, 3).reshape(batch_size, -1, self.N, self.N)
        branch_2 = Rk.reshape(batch_size, -1, self.N, self.N)

        return branch_1, branch_2


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class EncoderBranch(nn.Module):

    def __init__(
        self, in_channels: int, embed_dim: int = 256, downsample: bool = False
    ):
        super().__init__()
        # 1. Linear Channel Reduction Stem (2N^2 -> 512)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
        )
        # 2. Local Spatial Conv Block (Residual or ConvNeXt)
        self.spatial_block = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, embed_dim),
            nn.GELU(),
            ResBlock2D(embed_dim),
            ResBlock2D(embed_dim),
        )
        # 3. Downsampling block (conv with stride=2) if downsample is True
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.GroupNorm(16, embed_dim),
                nn.GELU(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        x = self.stem(x)
        x = self.spatial_block(x)
        x = self.downsample(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, query_src, key_val_src):
        attn_out, _ = self.attn(query=query_src, key=key_val_src, value=key_val_src)
        x = self.norm1(query_src + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DualBranchCrossAttention(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.cross_attn_12 = CrossAttentionBlock(embed_dim, num_heads)
        self.cross_attn_21 = CrossAttentionBlock(embed_dim, num_heads)

    def forward(self, F1, F2):
        B, C, H, W = F1.shape
        f1_seq = F1.flatten(2).transpose(1, 2)
        f2_seq = F2.flatten(2).transpose(1, 2)

        f1_out = self.cross_attn_12(query_src=f1_seq, key_val_src=f2_seq)
        f2_out = self.cross_attn_21(query_src=f2_seq, key_val_src=f1_seq)

        F1_updated = f1_out.transpose(1, 2).reshape(B, C, H, W)
        F2_updated = f2_out.transpose(1, 2).reshape(B, C, H, W)

        return F1_updated, F2_updated


class PhaseProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, N: int, phase_mask: torch.Tensor):
        super().__init__()
        self.N = N

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(
                128, 2, kernel_size=1
            ),  # Channel 0: Real (u), Channel 1: Imag (v)
        )

        self.register_buffer("aperture_mask", phase_mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Spatial upsampling if encoder/attention downsampled to (N/2, N/2)
        if x.shape[-2:] != (self.N, self.N):
            x = F.interpolate(
                x, size=(self.N, self.N), mode="bilinear", align_corners=False
            )

        # 2. Predict unconstrained Cartesian real/imag components
        uv = self.decoder(x)
        u, v = uv[:, 0:1], uv[:, 1:2]

        # 3. Enforce unit magnitude |p| = 1 and mask aperture
        norm = torch.sqrt(u**2 + v**2 + 1e-8)
        u_norm = (u / norm) * self.aperture_mask
        v_norm = (v / norm) * self.aperture_mask

        # 4. Form complex tensor output (B, N, N)
        return torch.complex(u_norm, v_norm).squeeze(1)


class KernelProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, N: int):
        super().__init__()
        self.kernel_grid_size = 2 * N - 1

        # Fuses Branch 1 and Branch 2 representation spaces
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1),  # Real and Imaginary components
        )

    def forward(self, F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
        # 1. Concatenate feature streams across channels
        F_fused = torch.cat([F1, F2], dim=1)

        # 2. Interpolate to relative difference grid dimensions (2N-1, 2N-1)
        F_rescaled = F.interpolate(
            F_fused,
            size=(self.kernel_grid_size, self.kernel_grid_size),
            mode="bilinear",
            align_corners=False,
        )

        # 3. Project to Cartesian output (B, 2, 2N-1, 2N-1)
        uv = self.decoder(F_rescaled)

        # 4. Return complex kernel tensor (B, 2N-1, 2N-1)
        return torch.complex(uv[:, 0], uv[:, 1])


def create_symmetric_aperture_mask(
    N: int, aperture_radius: float = 1.0
) -> torch.Tensor:
    # Pixel center offsets from physical grid center (N-1)/2
    center = (N - 1) / 2.0
    coords = (torch.arange(N, dtype=torch.float32) - center) / (N / 2.0)

    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    R = torch.sqrt(X**2 + Y**2)

    # Perfectly symmetric mask without even-N edge truncation
    mask = (R <= aperture_radius).float()
    return mask


class DualBranchPhaseNet(nn.Module):
    def __init__(
        self,
        N: int,
        embed_dim: int = 256,
        downsample: bool = False,
        aperture_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.N = N

        if aperture_mask is None:
            aperture_mask = create_symmetric_aperture_mask(N)

        self.folder = TensorFolder(N)
        in_channels = 2 * N * N

        self.enc_branch1 = EncoderBranch(in_channels, embed_dim, downsample)
        self.enc_branch2 = EncoderBranch(in_channels, embed_dim, downsample)

        self.cross_attn = DualBranchCrossAttention(embed_dim)

        self.head_p = PhaseProjectionHead(embed_dim, N, aperture_mask)
        self.head_q = PhaseProjectionHead(embed_dim, N, aperture_mask)

        self.head_o = KernelProjectionHead(embed_dim, N)

    def forward(
        self, R: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        branch_1, branch_2 = self.folder(R)

        F1 = self.enc_branch1(branch_1)
        F2 = self.enc_branch2(branch_2)

        F1_updated, F2_updated = self.cross_attn(F1, F2)

        p_pred = self.head_p(F1_updated)
        q_pred = self.head_q(F2_updated)

        # o_pred = self.head_o(F1_updated, F2_updated)
        o_pred = torch.zeros(
            (R.shape[0], 2 * self.N - 1, 2 * self.N - 1), device=R.device, dtype=R.dtype
        )

        return p_pred, q_pred, o_pred


class PhaseRetrievalLoss(nn.Module):
    def __init__(
        self, alpha_phase: float = 1.0, alpha_kernel: float = 1.0, eps: float = 1e-8
    ):
        super().__init__()
        self.alpha_phase = alpha_phase
        self.alpha_kernel = alpha_kernel
        self.eps = eps

    def complex_cosine_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Calculates phase-shift invariant complex inner-product distance."""
        # Scale-invariant complex inner product: 1 - |<pred, target>| / (||pred|| * ||target||)
        inner_prod = torch.sum(pred * torch.conj(target), dim=(-2, -1))
        norm_pred = torch.linalg.vector_norm(pred, dim=(-2, -1))
        norm_target = torch.linalg.vector_norm(target, dim=(-2, -1))

        similarity = torch.abs(inner_prod) / (norm_pred * norm_target + self.eps)
        return torch.mean(1.0 - similarity)

    def forward(
        self,
        p_pred: torch.Tensor,
        q_pred: torch.Tensor,
        o_pred: torch.Tensor,
        p_target: torch.Tensor,
        q_target: torch.Tensor,
        o_target: torch.Tensor,
    ) -> torch.Tensor:

        loss_p = self.complex_cosine_loss(p_pred, p_target)
        loss_q = self.complex_cosine_loss(q_pred, q_target)

        # Frobenius norm MSE for reconstructed object kernel
        loss_o = F.mse_loss(torch.view_as_real(o_pred), torch.view_as_real(o_target))

        total_loss = self.alpha_phase * (loss_p + loss_q) + self.alpha_kernel * loss_o
        return total_loss
