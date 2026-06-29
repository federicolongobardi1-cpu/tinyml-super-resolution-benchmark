import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.espcn import ESPCN


TRAIN_DIR = Path("/media/federico/4A21-0000/DIV2K_train_HR/DIV2K_train_HR")
PRETRAINED_WEIGHTS = Path("weights/espcn_x2_finetuned_224.pth.tar")
OUTPUT_WEIGHTS = Path("weights/espcn_x2_finetuned_224_stage2.pth.tar")

SCALE = 2
LR_SIZE = 224
HR_SIZE = LR_SIZE * SCALE

BATCH_SIZE = 4
EPOCHS = 5
STEPS_PER_EPOCH = 1000
LEARNING_RATE = 1e-6
SEED = 42


class DIV2KCropDataset(Dataset):
    def __init__(self, image_dir, steps_per_epoch):
        self.image_paths = sorted(Path(image_dir).glob("*.png"))
        self.steps_per_epoch = steps_per_epoch

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        path = random.choice(self.image_paths)

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")

        h, w = img.shape

        if h < HR_SIZE or w < HR_SIZE:
            img = cv2.resize(img, (max(w, HR_SIZE), max(h, HR_SIZE)), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape

        x0 = random.randint(0, w - HR_SIZE)
        y0 = random.randint(0, h - HR_SIZE)

        hr = img[y0:y0 + HR_SIZE, x0:x0 + HR_SIZE]
        lr = cv2.resize(hr, (LR_SIZE, LR_SIZE), interpolation=cv2.INTER_CUBIC)

        lr = lr.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        lr = torch.from_numpy(lr).unsqueeze(0)
        hr = torch.from_numpy(hr).unsqueeze(0)

        return lr, hr


def load_model():
    model = ESPCN(
        upscale_factor=SCALE,
        in_channels=1,
        out_channels=1,
    )

    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    return model


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = DIV2KCropDataset(TRAIN_DIR, STEPS_PER_EPOCH * BATCH_SIZE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model = load_model().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for step, (lr, hr) in enumerate(loader):
            lr = lr.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Step {step + 1}/{len(loader)} "
                    f"Loss {loss.item():.6f}"
                )

        mean_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} mean loss: {mean_loss:.6f}")

    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "scale": SCALE,
            "lr_size": LR_SIZE,
            "hr_size": HR_SIZE,
        },
        OUTPUT_WEIGHTS,
    )

    print(f"Saved fine-tuned weights to: {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()