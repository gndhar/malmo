import torch
import torch.nn as nn

from data_gen import RMDataset
from torch.utils.data import DataLoader

from zern import ZernikeAberration
from forward import Simulation
from rm import get_Rk_batched

from model import Model

N = 16
zern_n = 5
epochs = 50


train_dataset = RMDataset(N=2 * N, size=64, seed=42)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = RMDataset(N=2 * N, size=4, seed=420)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)


zern_gen = ZernikeAberration(N=N, zern_n=zern_n)
simulation = Simulation(N, dtype=torch.complex64)


model = Model(N, zern_gen.num_coefficients)
mse_criterion = nn.MSELoss()
optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)


def criterion(ab_in, ab_out, ab_pred_in, ab_pred_out):
    diff_ab_in = ab_pred_in * torch.conj(ab_in)
    diff_ab_out = ab_pred_out * torch.conj(ab_out)

    angle_in = torch.angle(diff_ab_in)
    angle_out = torch.angle(diff_ab_out)

    loss_in = torch.mean(angle_in**2)
    loss_out = torch.mean(angle_out**2)

    return loss_in + loss_out


# training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for c_in, c_out, obj in train_dataloader:
        with torch.no_grad():
            ab_in = zern_gen(c_in)
            ab_out = zern_gen(c_out)
            k_outs = simulation(ab_in, ab_out, obj)
            Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)

        optim.zero_grad()

        pred_in, pred_out = model(Rk)
        ab_pred_in = zern_gen(pred_in)
        ab_pred_out = zern_gen(pred_out)

        loss = criterion(ab_in, ab_out, ab_pred_in, ab_pred_out)
        loss.backward()
        optim.step()

        train_loss += loss.item()

    # Average training loss for the epoch
    avg_train_loss = train_loss / len(train_dataloader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0

    with torch.no_grad():  # Entire validation block skips gradient calculation
        for c_in, c_out, obj in val_dataloader:
            ab_in = zern_gen(c_in)
            ab_out = zern_gen(c_out)
            k_outs = simulation(ab_in, ab_out, obj)
            Rk = get_Rk_batched(k_in=simulation.k_in_cropped, k_outs=k_outs, N=N)

            pred_in, pred_out = model(Rk)
            ab_pred_in = zern_gen(pred_in)
            ab_pred_out = zern_gen(pred_out)

            loss = criterion(ab_in, ab_out, ab_pred_in, ab_pred_out)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)

    print(
        f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
    )
