import cv2
import torch
import numpy as np
from pathlib import Path

from models.espcn import ESPCN
from metrics import psnr

INPUT_PATH = Path("data/raw/test.png")
OUTPUT_PATH = Path("data/outputs/espcn_x2.png")
WEIGHTS_PATH = Path("weights/ESPCN_x2.pth.tar")

scale = 2

img = cv2.imread(str(INPUT_PATH))
if img is None:
    raise FileNotFoundError(f"Image not found: {INPUT_PATH}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape
new_h = h - (h % scale)
new_w = w - (w % scale)
gray = gray[:new_h, :new_w]

low = cv2.resize(
    gray,
    (new_w // scale, new_h // scale),
    interpolation=cv2.INTER_AREA
)

tensor = torch.from_numpy(low).float() / 255.0
tensor = tensor.unsqueeze(0).unsqueeze(0)

model = ESPCN(upscale_factor=2, in_channels=1, out_channels=1)

checkpoint = torch.load(str(WEIGHTS_PATH), map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

with torch.no_grad():
    output = model(tensor)

output = output.squeeze().numpy()
output = np.clip(output, 0, 1)
output = (output * 255).astype(np.uint8)

cv2.imwrite(str(OUTPUT_PATH), output)

print("Saved:", OUTPUT_PATH)
print("Original shape:", gray.shape)
print("Low shape:", low.shape)
print("Output shape:", output.shape)
print("ESPCN x2 PSNR:", psnr(gray, output))