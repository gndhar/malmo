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

# --- Configuration ---
ratio = 0.5
coeff_count = int(zern.cart.nk * ratio)
batch_size = 16
learning_rate = 1e-3
epochs = 100

# Since we don't have a static dataset length anymore, we define steps per epoch
samples_per_epoch = 1024
steps_per_epoch = samples_per_epoch // batch_size

# --- Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# --- Model & Optimizer Setup ---
model = ResNetEstimator(config.N, coeff_count).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Param count: {total_params:,}")
print(f"Trainable Param count: {trainable_params:,}\n")


# --- On-the-Fly GPU Data Generation ---
def generate_training_batch(batch_size, coeff_count, device):
    # 1. Generate coefficients on CPU using NumPy so zern.py can read them
    c_in_np = np.random.rand(batch_size, coeff_count).astype(np.float32)
    c_out_np = np.random.rand(batch_size, coeff_count).astype(np.float32)

    # 2. Vectorized GPU Simulation (Now handles batches)
    k_ins, k_outs = simulate_pt_vectorized(c_in_np, c_out_np, device)

    # 3. Generate Reflection Matrices natively on GPU
    R_k = generate_R_k_pt(k_ins, k_outs)

    # 4. Format for ResNet input (batch_size, N, N, N, N)
    N = config.N
    inputs = R_k.reshape(batch_size, N, N, N, N)

    # Target tensors move directly to device
    targets = torch.cat(
        (torch.tensor(c_in_np, device=device), torch.tensor(c_out_np, device=device)),
        dim=1,
    )

    return inputs, targets


# --- Training Loop ---
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    total_sim_time = 0.0
    total_train_time = 0.0

    pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}")

    for step in pbar:
        # 1. GPU Simulation Phase (Replaces DataLoader)
        t_sim_start = time.perf_counter()

        inputs, targets = generate_training_batch(batch_size, coeff_count, device)

        # Sync to get accurate timing for the simulation phase
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        sim_duration = time.perf_counter() - t_sim_start
        total_sim_time += sim_duration

        # 2. ML Training Phase
        t_train_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Sync to get accurate timing for the backprop phase
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()

        train_duration = time.perf_counter() - t_train_start
        total_train_time += train_duration

        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Sim/Train Ratio": f"{sim_duration/train_duration:.2f}x",
            }
        )

    # End of epoch summary
    avg_loss = running_loss / steps_per_epoch
    total_epoch_time = total_sim_time + total_train_time
    sim_percent = (
        (total_sim_time / total_epoch_time) * 100 if total_epoch_time > 0 else 0
    )

    print(f"\n--- Epoch {epoch+1} Summary ---")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Total Simulation Time: {total_sim_time:.2f}s")
    print(f"Total ML Training Time: {total_train_time:.2f}s")
    print(f"Simulation Overhead: {sim_percent:.1f}%\n")
