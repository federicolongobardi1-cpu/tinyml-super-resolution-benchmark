import torch

from models.espcn import ESPCN

checkpoint = torch.load(
    "weights/ESPCN_x2.pth.tar",
    map_location="cpu"
)

model = ESPCN(upscale_factor=2, in_channels=1, out_channels=1)

model.load_state_dict(checkpoint["state_dict"])

model.eval()

print("ESPCN x2 loaded correctly")
print("Best PSNR:", checkpoint["best_psnr"])
print("Epoch:", checkpoint["epoch"])