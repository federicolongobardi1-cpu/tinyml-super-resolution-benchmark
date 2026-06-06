import torch
from models.espcn import ESPCN

model = ESPCN(scale_factor=4, num_channels=3)
model.eval()

x = torch.rand(1, 3, 64, 64)

with torch.no_grad():
    y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)