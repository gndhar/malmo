import torch
import torch.nn as nn

from data_gen import RMDataset
from torch.utils.data import DataLoader

from zern import ZernikeAberration
from forward import Simulation
from rm import get_Rk_batched
from model import Model

# --- CONFIGURATION ---
N = 16
zern_n = 5
epochs = 50
pretrain_epochs = 50
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

# --- DATASETS & DATALOADERS ---
train_dataset = RMDataset(N=2 * N, size=1024, zern_n=zern_n, seed=42)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = RMDataset(N=2 * N, size=256, zern_n=zern_n, seed=420)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# --- MODULES (Moved to device) ---
zern_gen = ZernikeAberration(N=N, zern_n=zern_n).to(device)
simulation = Simulation(N, dtype=torch.complex64).to(device)
model = Model(N, zern_gen.num_coefficients).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

# --- LOSS FUNCTIONS ---
mse_loss = nn.MSELoss()


def criterion_pretrain(c_in, c_out, c_in_pred, c_out_pred):
    return mse_loss(c_in, c_in_pred) + mse_loss(c_out, c_out_pred)


def criterion(ab_in, ab_out, ab_pred_in, ab_pred_out):
    diff_ab_in = ab_pred_in * torch.conj(ab_in)
    diff_ab_out = ab_pred_out * torch.conj(ab_out)
    loss_in = torch.mean(torch.angle(diff_ab_in) ** 2)
    loss_out = torch.mean(torch.angle(diff_ab_out) ** 2)
    return loss_in + loss_out


# --- HELPER FUNCTION FOR TRAINING & VALIDATION ---
def run_epoch(dataloader, is_train, criterion_func, is_pretrain=False):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for c_in, c_out, obj in dataloader:
            # Move data to device
            c_in, c_out, obj = c_in.to(device), c_out.to(device), obj.to(device)

            with torch.no_grad():
                ab_in = zern_gen(c_in)
                ab_out = zern_gen(c_out)
                k_outs = simulation(ab_in, ab_out, obj)
                Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)

            if is_train:
                optimizer.zero_grad()

            pred_in, pred_out = model(Rk)

            if is_pretrain:
                loss = criterion_func(c_in, c_out, pred_in, pred_out)
            else:
                ab_pred_in = zern_gen(pred_in)
                ab_pred_out = zern_gen(pred_out)
                loss = criterion_func(ab_in, ab_out, ab_pred_in, ab_pred_out)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)


# --- EXECUTION ---
print("--- Starting Pre-training ---")
for epoch in range(pretrain_epochs):
    train_loss = run_epoch(
        train_dataloader,
        is_train=True,
        criterion_func=criterion_pretrain,
        is_pretrain=True,
    )
    val_loss = run_epoch(
        val_dataloader,
        is_train=False,
        criterion_func=criterion_pretrain,
        is_pretrain=True,
    )
    print(
        f"PT Epoch [{epoch+1}/{pretrain_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

print("\n--- Starting Main Training ---")
for epoch in range(epochs):
    train_loss = run_epoch(
        train_dataloader, is_train=True, criterion_func=criterion, is_pretrain=False
    )
    val_loss = run_epoch(
        val_dataloader, is_train=False, criterion_func=criterion, is_pretrain=False
    )
    print(
        f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )
