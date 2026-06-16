"""
train_v4.py
-----------
Trains OrthogonalAberrationNet on complex field phasors rather than coefficient vectors.
Features:
  - 4096 samples per epoch (to match the larger generated dataset).
  - Model predicts coefficients, which are internally projected onto a Gram-Schmidt 
    orthogonalized discrete Zernike basis.
  - Loss is applied directly to the reconstructed complex phasors, eliminating
    the residual map and forcing the network to learn actual spatial shapes.
"""

import os
import time
import shutil
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from config import config
import zern
from resnet_ortho import OrthogonalAberrationNet

from forward_pt import simulate_pt_vectorized
from reflection_pt import generate_R_k_pt
from obj import get_batch_objs

# ── Hyper-parameters ──────────────────────────────────────────────────────────
RATIO       = 0.5
coeff_count = int(zern.cart.nk * RATIO)
batch_size  = 4
lr          = 1e-3

epochs      = 1000

samples_per_epoch = 1024
steps_per_epoch   = samples_per_epoch // batch_size

N = config.N
N2 = N // 2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Model & Checkpoint Logic ──────────────────────────────────────────────────
model     = OrthogonalAberrationNet(N, coeff_count, feat_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
use_amp   = device.type == "cuda"
scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}\n")

start_epoch = 0
CKPT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "model_final_v4.pt")

if os.path.isfile(CKPT_PATH):
    print(f"Found existing checkpoint at {CKPT_PATH}")
    # Backup the checkpoint
    bak_path = CKPT_PATH + ".bak"
    shutil.copy(CKPT_PATH, bak_path)
    print(f"Created backup at {bak_path}")
    
    # Load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # If resuming from the same experiment
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if "scaler_state_dict" in ckpt and use_amp:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if "epoch" in ckpt:
        # Note: if the ckpt saved 'epoch' as the total epochs, this logic handles it
        # depending on how it was saved. Let's assume it was saved as the current completed epoch.
        start_epoch = ckpt.get("current_epoch", ckpt["epoch"]) 
        # If the saved 'epoch' equals total epochs (like our old save code), 
        # we might just want to check if start_epoch >= epochs and adjust.
        if start_epoch >= epochs:
            print("Checkpoint epoch >= target epochs. Adjusting or continuing...")

    print(f"Resuming training from epoch {start_epoch}\n")

def augment_obj_batch(obj_batch: torch.Tensor) -> torch.Tensor:
    B = obj_batch.shape[0]
    for b in range(B):
        img = obj_batch[b]
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[-1])
        if torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[-2])
        k = int(torch.randint(0, 4, (1,)).item())
        if k > 0:
            img = torch.rot90(img, k=k, dims=[-2, -1])
        img = img * (0.6 + 0.4 * torch.rand(1).item())
        obj_batch[b] = img
    return obj_batch

def generate_training_batch(batch_size: int, coeff_count: int, device):
    rng = np.random.default_rng()

    c_in_np  = rng.uniform(-1.0, 1.0, (batch_size, coeff_count)).astype(np.float32)
    c_out_np = rng.uniform(-1.0, 1.0, (batch_size, coeff_count)).astype(np.float32)
    
    # Pad to full Zernike order to generate ground truth phases
    c_in_full  = np.zeros((batch_size, zern.cart.nk), dtype=np.float32)
    c_out_full = np.zeros((batch_size, zern.cart.nk), dtype=np.float32)
    c_in_full[:, :coeff_count]  = c_in_np
    c_out_full[:, :coeff_count] = c_out_np

    obj_batch = get_batch_objs(batch_size, rng=rng, device=device)
    obj_batch = augment_obj_batch(obj_batch)

    k_ins, k_outs = simulate_pt_vectorized(
        c_in_np, c_out_np, device=device, obj_batch=obj_batch, rng=rng
    )
    R_k    = generate_R_k_pt(k_ins, k_outs)
    inputs = R_k.reshape(batch_size, N, N, N, N)

    # Generate ground truth aberration maps (cropped to center)
    in_abb_np  = np.stack([zern.generate_abberations(c) for c in c_in_full])
    out_abb_np = np.stack([zern.generate_abberations(c) for c in c_out_full])
    
    phi_in_gt  = np.angle(in_abb_np[:, N2:N2+N, N2:N2+N])
    phi_out_gt = np.angle(out_abb_np[:, N2:N2+N, N2:N2+N])
    
    # Create target complex phasors
    E_in_gt  = torch.tensor(np.exp(1j * phi_in_gt), device=device, dtype=torch.cfloat)
    E_out_gt = torch.tensor(np.exp(1j * phi_out_gt), device=device, dtype=torch.cfloat)

    targets = (E_in_gt, E_out_gt)
    return inputs, targets

# ── Training loop ─────────────────────────────────────────────────────────────
if start_epoch >= epochs:
    print(f"Model already trained to epoch {start_epoch}. Exiting.")
    import sys
    sys.exit(0)

model.train()
epoch = start_epoch - 1  # ensure it's defined even if the loop fails

for epoch in range(start_epoch, epochs):
    running_loss   = 0.0
    total_sim_time = 0.0
    total_trn_time = 0.0

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1:03d}/{epochs}")

    for step in pbar:
        t0 = time.perf_counter()
        inputs, (E_in_gt, E_out_gt) = generate_training_batch(batch_size, coeff_count, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_sim_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            _, phi_in_pred, phi_out_pred = model(inputs)
            
            # Predict complex phasors
            E_in_pred  = torch.complex(torch.cos(phi_in_pred), torch.sin(phi_in_pred))
            E_out_pred = torch.complex(torch.cos(phi_out_pred), torch.sin(phi_out_pred))
            
            # MSE on complex phasors
            loss_in  = torch.mean(torch.abs(E_in_pred - E_in_gt)**2)
            loss_out = torch.mean(torch.abs(E_out_pred - E_out_gt)**2)
            loss = loss_in + loss_out

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_trn_time += time.perf_counter() - t1
        running_loss   += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    avg_loss = running_loss / steps_per_epoch
    print(
        f"  Loss={avg_loss:.4f}  "
        f"sim={total_sim_time:.1f}s  trn={total_trn_time:.1f}s  "
        f"lr={scheduler.get_last_lr()[0]:.2e}"
    )

# ── Save checkpoint ───────────────────────────────────────────────────────────
CKPT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "model_final_v4.pt")
os.makedirs(CKPT_DIR, exist_ok=True)

torch.save(
    {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if use_amp else None,
        "coeff_count":          coeff_count,
        "epoch":                epochs,
        "current_epoch":        epoch + 1,
        "final_loss":           avg_loss,
        "arch":                 "OrthogonalAberrationNet",
        "N":                    N,
        "feat_dim":             256,
    },
    CKPT_PATH,
)
print(f"\nCheckpoint saved → {CKPT_PATH}")
