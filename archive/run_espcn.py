import cv2
import torch
import numpy as np

from models.espcn import ESPCN


# -----------------------------
# load image
# -----------------------------
img = cv2.imread("data/raw/test.png")

if img is None:
    raise FileNotFoundError("Image not found")


# -----------------------------
# create low resolution image
# -----------------------------
scale = 4

h, w = img.shape[:2]

low = cv2.resize(
    img,
    (w // scale, h // scale),
    interpolation=cv2.INTER_AREA
)


# -----------------------------
# convert image to tensor
# -----------------------------
low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)

tensor = torch.from_numpy(low_rgb).float()

tensor = tensor.permute(2, 0, 1)

tensor = tensor.unsqueeze(0)

tensor = tensor / 255.0


# -----------------------------
# load model
# -----------------------------
model = ESPCN(scale_factor=scale)

model.eval()


# -----------------------------
# inference
# -----------------------------
with torch.no_grad():
    output = model(tensor)


# -----------------------------
# tensor -> image
# -----------------------------
output = output.squeeze(0)

output = output.permute(1, 2, 0)

output = output.numpy()

output = np.clip(output, 0, 1)

output = (output * 255).astype(np.uint8)

output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


# -----------------------------
# save result
# -----------------------------
cv2.imwrite("data/outputs/espcn_output.png", output_bgr)

print("ESPCN output saved")