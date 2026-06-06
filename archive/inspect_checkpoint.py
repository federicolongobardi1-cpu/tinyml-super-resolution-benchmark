import torch

checkpoint = torch.load(
    "weights/ESPCN_x2.pth.tar",
    map_location="cpu"
)

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nKEYS:")
    for key in checkpoint.keys():
        print(key)