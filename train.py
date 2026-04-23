# train.py
from config import config
import torch
from torch import nn
import zern
import numpy as np
import forward_sim
import reflection_matrix

ratio = 0.5
coeff_count = int(zern.cart.nk * ratio)


# X.shape = (batch_size, config.N, config.N, config.N, config.N)
# out.shape = (batch_size, 2*coeff_count)
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2 * config.N**4, 2 * coeff_count)

    def forward(self, x: torch.Tensor):
        x = x.reshape(x.shape[0], -1)
        y = torch.cat((x.real, x.imag), dim=1)
        # print(y.shape)

        return self.linear(y)


# training loop

from torch.utils.data import Dataset, DataLoader


class ForwardSimDataset(Dataset):
    def __init__(self, num_samples, coeff_count):
        self.num_samples = num_samples
        self.coeff_count = coeff_count

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        c_in = list(np.random.random(self.coeff_count))
        c_out = list(np.random.random(self.coeff_count))

        s_in, s_out = forward_sim.simulate(c_in, c_out)
        *_, R_k = reflection_matrix.generate_R(s_in, s_out)
        # print(R_k.shape)
        N = config.N
        R_k = R_k.reshape((N, N, N, N))
        # R_k: np.ndarray
        target = torch.cat((torch.tensor(c_in), torch.tensor(c_out)), dim=0)
        return torch.tensor(R_k, dtype=torch.cfloat), target.float()


batch_size = 4
learning_rate = 1e-3
epochs = 1

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
from resnet import ResNetEstimator

model = ResNetEstimator(config.N, coeff_count).to(device)

total_params = sum(p.numel() for p in model.parameters())

# Calculate trainable parameters (those that require gradients)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Param count: {total_params:,}")
print(f"Trainable Param count: {trainable_params:,}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset = ForwardSimDataset(1000, coeff_count)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training loop
model.train()
from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
