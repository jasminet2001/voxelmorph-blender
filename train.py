import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from model import VoxelMorphNet

shape_A = np.zeros((32, 32, 32))
shape_A[8:24, 8:24, 8:24] = 1

shape_B = np.zeros((32, 32, 32))
shape_B[10:26, 10:26, 8:24] = 1

a = torch.tensor(shape_A, dtype=torch.float32)
b = torch.tensor(shape_B, dtype=torch.float32)

x = torch.stack([a, b], dim=0).unsqueeze(0)

# The training loop
model = VoxelMorphNet()
# measuring the loss fucntion and optimizing it afterwards
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def warp(shape, field):
    grid = torch.nn.functional.affine_grid(
        torch.eye(3, 4).unsqueeze(0),
        shape.shape,
        align_corners=True
    )
    grid = grid + field.permute(0, 2, 3, 4, 1)
    return F.grid_sample(shape, grid, align_corners=True)


loss_fn = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    field = model(x)
    warped_a = warp(a.unsqueeze(0).unsqueeze(0), field)
    loss = loss_fn(warped_a, b.unsqueeze(0).unsqueeze(0))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

field_np = field.detach().numpy()
np.save("displacement_field.npy", field_np)
