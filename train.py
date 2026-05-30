"""
train.py  (malmo_dataset version)
-----------------------------------
On-the-fly training loop for the aberration-correction network.

Differences from the original malmo/train.py:
  1. Each batch samples a DIFFERENT object image from the scene pool (5k PNGs).
  2. In-batch augmentation is applied to the object images before simulation.
  3. Zernike coefficient range follows Seong et al. 2025: U(-1, 1).
  4. Aberration order is randomly chosen per batch (10 to zern.cart.nk).
"""

import time
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from config import config
import zern
from resnet import ResNetEstimator

from forward_pt import simulate_pt_vectorized
from reflection_pt import generate_R_k_pt
from obj import get_batch_objs

#Configuration
ratio = 0.5
coeff_count = int(zern.cart.nk * ratio)
batch_size = 16
learning_rate = 1e-3
epochs = 100

samples_per_epoch = 1024
steps_per_epoch   = samples_per_epoch // batch_size

#Device Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Model, Optimizer & AMP scaler
model     = ResNetEstimator(config.N, coeff_count).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# AMP: runs ResNet in float16 on CUDA — halves activation memory
use_amp   = device.type == "cuda"
scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Param count:     {total_params:,}")
print(f"Trainable Param count: {trainable_params:,}\n")


#Augmentation
def augment_obj_batch(obj_batch: torch.Tensor) -> torch.Tensor:
    """
    Lightweight on-GPU augmentation applied to a batch of object images.

    obj_batch : (B, 2N, 2N) complex tensor
    Returns   : (B, 2N, 2N) complex tensor

    Augmentations (each applied independently per sample):
      - Random horizontal flip     (p=0.5)
      - Random vertical flip       (p=0.5)
      - Random 90° rotation        (0, 90, 180, 270°)
      - Random intensity scale     U(0.6, 1.0)
    """
    B = obj_batch.shape[0]

    for b in range(B):
        img = obj_batch[b]  # (2N, 2N)

        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[-1])   # horizontal
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[-2])   # vertical

        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            img = torch.rot90(img, k=k, dims=[-2, -1])

        scale = 0.6 + 0.4 * torch.rand(1, device=obj_batch.device).item()
        img = img * scale

        obj_batch[b] = img

    return obj_batch


# On-the-Fly GPU Data Generation

def generate_training_batch(batch_size: int, coeff_count: int, device: torch.device):
    """
    Generate one training batch entirely on-the-fly.

    Pipeline
    --------
    1. Sample random Zernike coefficients in U(-1, 1)  [matches Seong et al.]
    2. Sample `batch_size` different object images from the scene pool
    3. Apply random augmentation to the object batch
    4. Run batched GPU forward simulation → k_ins, k_outs
    5. Build reflection matrices R_k on GPU
    6. Return (inputs, targets) where targets are the aberration coefficients
    """
    rng = np.random.default_rng()

    # 1. Random coefficients in U(-1, 1)  (paper uses this range)
    c_in_np  = rng.uniform(-1.0, 1.0, size=(batch_size, coeff_count)).astype(np.float32)
    c_out_np = rng.uniform(-1.0, 1.0, size=(batch_size, coeff_count)).astype(np.float32)

    # 2. Sample different object images for each item in the batch
    obj_batch = get_batch_objs(batch_size, rng=rng, device=device)  # (B, 2N, 2N) cfloat

    # 3. Augment objects
    obj_batch = augment_obj_batch(obj_batch)

    # 4. Forward simulation on GPU
    k_ins, k_outs = simulate_pt_vectorized(
        c_in_np, c_out_np, device=device, obj_batch=obj_batch, rng=rng
    )

    # 5. Build reflection matrices on GPU
    R_k = generate_R_k_pt(k_ins, k_outs)

    # 6. Reshape for ResNet: (batch_size, N, N, N, N)
    N      = config.N
    inputs = R_k.reshape(batch_size, N, N, N, N)

    targets = torch.cat(
        (
            torch.tensor(c_in_np,  device=device),
            torch.tensor(c_out_np, device=device),
        ),
        dim=1,
    )

    return inputs, targets


# Training Loop
model.train()

for epoch in range(epochs):
    running_loss   = 0.0
    total_sim_time = 0.0
    total_train_time = 0.0

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}")

    for step in pbar:
        # Simulation Phase
        t_sim_start = time.perf_counter()

        inputs, targets = generate_training_batch(batch_size, coeff_count, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        sim_duration     = time.perf_counter() - t_sim_start
        total_sim_time  += sim_duration

        # ML Training Phase
        t_train_start = time.perf_counter()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        train_duration     = time.perf_counter() - t_train_start
        total_train_time  += train_duration
        running_loss      += loss.item()

        pbar.set_postfix(
            {
                "Loss":            f"{loss.item():.4f}",
                "Sim/Train Ratio": f"{sim_duration / max(train_duration, 1e-9):.2f}x",
            }
        )

    # Epoch Summary
    avg_loss        = running_loss / steps_per_epoch
    total_epoch_time = total_sim_time + total_train_time
    sim_percent     = (total_sim_time / total_epoch_time * 100) if total_epoch_time > 0 else 0

    print(f"\n--- Epoch {epoch + 1} Summary ---")
    print(f"Loss:                    {avg_loss:.4f}")
    print(f"Total Simulation Time:   {total_sim_time:.2f}s")
    print(f"Total ML Training Time:  {total_train_time:.2f}s")
    print(f"Simulation Overhead:     {sim_percent:.1f}%\n")
